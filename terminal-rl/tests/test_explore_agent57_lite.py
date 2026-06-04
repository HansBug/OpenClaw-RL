from __future__ import annotations

import sys
from pathlib import Path

TERMINAL_RL_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = TERMINAL_RL_DIR.parent
if str(TERMINAL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(TERMINAL_RL_DIR))
if str(ROOT_DIR / "slime") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "slime"))

import explore_agent57_lite as a57


def _reset_local_agent57_state():
    a57._LOCAL_COUNTS.clear()
    a57._LOCAL_ARM_EVENTS.clear()
    a57._LOCAL_TRAJ_SEEN = 0


def test_agent57_config_defaults_preserve_additive_mode(monkeypatch):
    for name in (
        "EXPLORE_AGENT57_COMBINE_MODE",
        "EXPLORE_AGENT57_NGU_MOD_CLIP",
        "EXPLORE_AGENT57_NGU_EPISODIC_SOURCE",
        "EXPLORE_AGENT57_MAX_BONUS",
    ):
        monkeypatch.delenv(name, raising=False)

    config = a57.config_from_env()

    assert config.combine_mode == "add"
    assert config.ngu_mod_clip == 5.0
    assert config.ngu_episodic_source == "signature_intrinsic"
    assert config.max_bonus == 0.0


def test_ngu_lite_bonus_uses_product_and_clamp(monkeypatch):
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_COMBINE_MODE", "ngu_lite")
    monkeypatch.setenv("EXPLORE_AGENT57_ARM_BETAS", "0.02")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_COEF", "0.5")
    monkeypatch.setenv("EXPLORE_AGENT57_NGU_MOD_CLIP", "3")
    monkeypatch.setenv("EXPLORE_AGENT57_MAX_BONUS", "0.05")
    config = a57.config_from_env()

    metrics = a57.compute_ngu_lite_bonus(
        config=config,
        arm_id=0,
        episodic_novelty=10.0,
        lifelong_raw=5.0,
        lifelong_eligible=True,
    )

    assert metrics["explore_agent57_ngu_life_mod"] == 3.0
    assert metrics["explore_agent57_ngu_bonus_unclipped"] == 0.3
    assert metrics["explore_agent57_ngu_bonus"] == 0.05
    assert metrics["explore_agent57_bonus_clipped"] == 1.0


def test_ngu_lite_bonus_stays_zero_when_lifelong_not_eligible(monkeypatch):
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_COMBINE_MODE", "ngu_lite")
    monkeypatch.setenv("EXPLORE_AGENT57_ARM_BETAS", "0.02")
    config = a57.config_from_env()

    metrics = a57.compute_ngu_lite_bonus(
        config=config,
        arm_id=0,
        episodic_novelty=10.0,
        lifelong_raw=1.0,
        lifelong_eligible=False,
    )

    assert metrics["explore_agent57_ngu_bonus"] == 0.0
    assert metrics["explore_agent57_ngu_episodic"] == 10.0


def test_ucb_min_per_arm_prioritizes_under_sampled_arms(monkeypatch):
    _reset_local_agent57_state()
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_CONTROLLER", "ucb")
    monkeypatch.setenv("EXPLORE_AGENT57_K", "4")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "local")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_C", "0")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_MIN_PER_ARM", "2")
    monkeypatch.setenv("EXPLORE_AGENT57_KEEP_BASELINE", "1")
    config = a57.config_from_env()

    for arm_id in (0, 1):
        for _ in range(2):
            a57.record_arm_event(
                config=config,
                arm_id=arm_id,
                base_score=1.0,
                final_score=1.0,
                status="completed",
                parse_error_count=0,
                bonus=0.0,
                dataset="seta",
            )

    arms = a57.assign_group_arms(4, dataset="seta")

    assert arms[:3] == [0, 2, 3]


def test_ucb_dataset_aware_uses_normalized_base_reward(monkeypatch):
    _reset_local_agent57_state()
    monkeypatch.setenv("EXPLORE_AGENT57_LITE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_CONTROLLER", "ucb")
    monkeypatch.setenv("EXPLORE_AGENT57_K", "3")
    monkeypatch.setenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "local")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_C", "0")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_VALUE", "normalized_base")
    monkeypatch.setenv("EXPLORE_AGENT57_UCB_DATASET_AWARE", "1")
    monkeypatch.setenv("EXPLORE_AGENT57_KEEP_BASELINE", "0")
    config = a57.config_from_env()

    for dataset, scores in (
        ("seta", {0: 0.0, 1: 1.0, 2: 0.0}),
        ("agentharm", {0: -1.0, 1: -1.0, 2: 1.0}),
    ):
        for arm_id, score in scores.items():
            a57.record_arm_event(
                config=config,
                arm_id=arm_id,
                base_score=score,
                final_score=score,
                status="completed",
                parse_error_count=0,
                bonus=0.0,
                dataset=dataset,
            )

    assert a57.assign_group_arms(1, dataset="seta") == [1]
    assert a57.assign_group_arms(1, dataset="agentharm") == [2]
