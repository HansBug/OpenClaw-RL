from __future__ import annotations

from types import SimpleNamespace

import sys
from pathlib import Path

TERMINAL_RL_DIR = Path(__file__).resolve().parents[1]
if str(TERMINAL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(TERMINAL_RL_DIR))

import reward_postprocess


class DummySample:
    def __init__(
        self,
        *,
        group_index: int,
        index: int,
        score: float,
        intrinsic: float,
        beta: float,
        trust: float = 1.0,
    ) -> None:
        self.group_index = group_index
        self.index = index
        self.reward = {
            "score": score,
            "raw_score": score,
            "base_score": score,
            "explore_agent57_intrinsic_signal": intrinsic,
            "explore_agent57_beta": beta,
            "explore_agent57_trust": trust,
        }


def test_dual_stream_advantage_adds_group_normalized_intrinsic(monkeypatch):
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_ENABLED", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_MODE", "dual_stream")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_LAMBDA", "0.2")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE", "normalized_beta")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_CLIP", "0")
    args = SimpleNamespace(
        reward_key="score",
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        dynamic_history=False,
    )
    samples = [
        DummySample(group_index=0, index=0, score=1.0, intrinsic=0.0, beta=0.01),
        DummySample(group_index=0, index=1, score=1.0, intrinsic=1.0, beta=0.02),
    ]

    raw, adjusted = reward_postprocess.post_process_rewards(args, samples)

    assert raw == [1.0, 1.0]
    assert adjusted == [-0.05, 0.1]
    assert samples[0].reward["explore_post_norm_bonus_mode"] == "dual_stream"
    assert samples[1].reward["explore_post_norm_base_reward"] == 0.0
    assert samples[1].reward["explore_post_norm_intrinsic_value"] == 1.0
    assert samples[1].reward["explore_post_norm_intrinsic_advantage"] == 0.5
    assert samples[1].reward["explore_post_norm_trust"] == 1.0
    assert samples[1].reward["explore_post_norm_adjusted_reward"] == 0.1


def test_component_postnorm_mode_remains_backward_compatible(monkeypatch):
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS", "1")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_ENABLED", "1")
    monkeypatch.delenv("EXPLORE_ADVANTAGE_BONUS_MODE", raising=False)
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_COMPONENTS", "explore_intrinsic_scaled")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_COEF", "1.0")
    monkeypatch.setenv("EXPLORE_ADVANTAGE_BONUS_CLIP", "0.25")
    args = SimpleNamespace(
        reward_key="score",
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        dynamic_history=False,
    )
    sample = DummySample(group_index=0, index=0, score=1.0, intrinsic=0.0, beta=0.0)
    sample.reward["explore_intrinsic_scaled"] = 0.5

    _, adjusted = reward_postprocess.post_process_rewards(args, [sample])

    assert adjusted == [0.25]
    assert sample.reward["explore_post_norm_bonus_mode"] == "component"
    assert sample.reward["explore_post_norm_base_reward"] == 0.0
    assert sample.reward["explore_post_norm_adjusted_reward"] == 0.25
