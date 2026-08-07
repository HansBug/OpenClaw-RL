"""Guards for the mode B aligned Harbor adapter.

The adapter's value is that its defaults reproduce the training-time harness, so
the evals recorded in docs/HARBOR_CAMEL_MODE_B_zh.md stay comparable. Silently
changing one of those defaults would not break anything at runtime, it would just
make every future number incomparable to the recorded ones -- which is exactly the
kind of drift a test has to catch.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TERMINAL_RL = ROOT / "terminal-rl"
MODE_B = TERMINAL_RL / "eval" / "mode_b_aligned"
ADAPTER_DIR = MODE_B / "adapter"
ADAPTER_PY = ADAPTER_DIR / "openclaw_camel_adapter.py"
LAUNCHERS = sorted((MODE_B / "launchers").glob("*.sh"))

# Importing the adapter pulls in harbor, camel and transformers. Skip rather than
# fail where those are absent, so the rest of the suite stays runnable.
pytest.importorskip("harbor", reason="harbor is required to import the adapter")
pytest.importorskip("camel", reason="camel-ai is required to import the adapter")

if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

import openclaw_camel_adapter as adapter_module  # noqa: E402

OpenClawCamelAgent = adapter_module.OpenClawCamelAgent


def _agent(**kwargs):
    return OpenClawCamelAgent(
        logs_dir=Path(tempfile.mkdtemp()),
        sglang_served_name="test-served-name",
        hf_model_dir="/nonexistent/hf-dir",
        **kwargs,
    )


def test_terminal_rl_root_resolves_from_the_adapter_location():
    """The adapter must work from any checkout and cwd, without env setup."""
    resolved = Path(adapter_module._resolve_terminal_rl_dir())
    assert resolved == TERMINAL_RL
    assert (resolved / "agent" / "camel_agent.py").is_file()


def test_terminal_rl_root_override_reports_the_bad_path(tmp_path, monkeypatch):
    """A wrong OPENCLAW_TERMINAL_RL_DIR must fail loudly, not as ModuleNotFoundError."""
    monkeypatch.setenv("OPENCLAW_TERMINAL_RL_DIR", str(tmp_path))
    with pytest.raises(RuntimeError) as excinfo:
        adapter_module._resolve_terminal_rl_dir()
    message = str(excinfo.value)
    assert str(tmp_path) in message
    assert "OPENCLAW_TERMINAL_RL_DIR" in message


@pytest.mark.parametrize(
    "kwargs, missing",
    [
        ({}, "sglang_served_name"),
        ({"sglang_served_name": "some-model"}, "hf_model_dir"),
    ],
)
def test_checkpoint_identity_kwargs_are_required(kwargs, missing):
    """A wrong served name or tokenizer dir evaluates the wrong thing silently."""
    with pytest.raises(ValueError, match=missing):
        OpenClawCamelAgent(logs_dir=Path(tempfile.mkdtemp()), **kwargs)


def test_aligned_knob_defaults_match_the_recorded_evals():
    """Pins knobs 5-14 of the alignment table in docs/HARBOR_CAMEL_MODE_B_zh.md."""
    agent = _agent()
    assert agent.max_iteration == 10
    assert agent.max_parse_errors == 3
    assert agent.temperature == 1.0
    assert agent.top_p == 1.0
    assert agent.top_k == -1
    assert agent.max_new_tokens == 8192
    assert agent.max_total_tokens == 16384
    assert agent.rollout_skip_special_tokens is False
    assert agent.rollout_seed == 42
    assert agent.tool_call_parser == "qwen25"
    assert agent.non_think_mode is False


def test_sglang_url_is_normalised_to_the_generate_endpoint():
    """Callers pass a server root; the client needs the /generate path."""
    assert _agent(sglang_url="http://127.0.0.1:30000").sglang_url.endswith("/generate")
    assert _agent(sglang_url="http://127.0.0.1:30000/generate").sglang_url.count("/generate") == 1


def test_adapter_reports_a_stable_identity():
    """Harbor writes these into the job manifest, so results stay attributable."""
    assert OpenClawCamelAgent.name() == "openclaw-camel-agent"
    assert _agent().version() == "0.1.0"


def test_launcher_scripts_exist():
    """Without this, the parametrized checks below would vacuously pass on an empty glob."""
    assert {p.name for p in LAUNCHERS} == {"launch_sglang.sh", "run_harbor_eval.sh"}


@pytest.mark.parametrize("script", LAUNCHERS, ids=lambda p: p.name)
def test_launchers_parse_and_are_executable(script):
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert script.stat().st_mode & 0o111, f"{script.name} is not executable"


@pytest.mark.parametrize(
    "path",
    [ADAPTER_PY] + LAUNCHERS,
    ids=lambda p: p.name,
)
def test_runtime_files_carry_no_site_specific_absolute_paths(path):
    """Site paths belong in docs and env vars, never in code the next site runs."""
    offenders = [
        f"{path.name}:{lineno}: {line.strip()}"
        for lineno, line in enumerate(path.read_text().splitlines(), start=1)
        if "/mnt/shared-storage-user/" in line
        or "/mnt/data/deepghs/" in line
        or "/nfs/eval_results/" in line
    ]
    assert not offenders, "site-specific absolute paths must not be hardcoded:\n" + "\n".join(offenders)
