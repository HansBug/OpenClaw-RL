from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TERMINAL_RL = ROOT / "terminal-rl"


def test_rollout_configs_default_to_camel_agent():
    for name in ("rollout_qwen3.yaml", "rollout_qwen3_think.yaml"):
        cfg = yaml.safe_load((TERMINAL_RL / "configs" / name).read_text())
        assert cfg["harness_option"] == "camel-agent"


def test_training_script_routes_harness_without_polluting_camel_runtime():
    script = (TERMINAL_RL / "terminal-rl_qwen3-8b_pu.sh").read_text()
    assert 'HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"' in script
    assert 'cfg["harness_option"] = harness_option' in script
    assert 'if [[ "${HARNESS_OPTION}" == "a3s-code" && "${DRY_RUN}" != "1" ]]' in script
    assert 'if [[ "${HARNESS_OPTION}" == "a3s-code" ]]; then' in script
    assert '\\"HARNESS_OPTION\\": \\"${HARNESS_OPTION}\\"' in script
    assert '-- "${TRAIN_PYTHON}" -u "${SLIME_DIR}/train_async.py"' in script
