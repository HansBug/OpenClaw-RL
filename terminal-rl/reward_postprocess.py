from __future__ import annotations

import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.4f", name, raw, default)
        return default


def _env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    return default if raw is None else raw.strip()


def _reward_value(args: Any, sample: Any) -> float:
    reward = getattr(sample, "reward", None)
    key = getattr(args, "reward_key", None)
    if key:
        if not isinstance(reward, dict):
            return 0.0
        return float(reward.get(key, 0.0) or 0.0)
    return float(reward or 0.0)


def _component_value(sample: Any, key: str) -> float:
    reward = getattr(sample, "reward", None)
    if not isinstance(reward, dict):
        return 0.0
    value = reward.get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _sync_reward_aliases(
    reward: dict[str, Any] | None,
    *,
    total_reward: float | None = None,
    extra_exploration_reward: float = 0.0,
) -> None:
    if not isinstance(reward, dict):
        return
    total = reward.get("score") if total_reward is None else total_reward
    raw = reward.get("raw_score", total)
    task = reward.get("base_score", raw)
    exploration = float(reward.get("explore_total_bonus", 0.0) or 0.0) + extra_exploration_reward
    reward["raw_reward"] = raw
    reward["task_reward"] = task
    reward["exploration_reward"] = exploration
    reward["total_reward"] = total


def _normalize_values(values: list[float], use_std: bool) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    centered = [v - mean for v in values]
    if not use_std:
        return centered
    if len(values) <= 1:
        return [0.0 for _ in values]
    # Match torch.std default semantics used by slime: unbiased sample std.
    var = sum(v * v for v in centered) / max(1, len(values) - 1)
    std = math.sqrt(max(var, 0.0))
    return [v / (std + 1e-6) for v in centered]


def _sample_group_key(sample: Any) -> int:
    return int(sample.group_index) if getattr(sample, "group_index", None) is not None else -1


def _sample_traj_key(sample: Any, sample_idx: int) -> tuple[int, int]:
    group_idx = _sample_group_key(sample)
    traj_idx = int(sample.index) if getattr(sample, "index", None) is not None else sample_idx
    return group_idx, traj_idx


def _group_normalize_sample_values(
    args: Any,
    samples: list[Any],
    values: list[float],
) -> list[float]:
    use_std = bool(getattr(args, "grpo_std_normalization", False))
    if getattr(args, "dynamic_history", False):
        value_by_key: dict[tuple[int, int], float] = {}
        group_to_keys: dict[int, list[tuple[int, int]]] = {}
        key_by_sample: list[tuple[int, int]] = []
        for i, sample in enumerate(samples):
            key = _sample_traj_key(sample, i)
            key_by_sample.append(key)
            if key not in value_by_key:
                value_by_key[key] = float(values[i])
                group_to_keys.setdefault(key[0], []).append(key)

        normalized_by_key: dict[tuple[int, int], float] = {}
        for keys in group_to_keys.values():
            vals = _normalize_values([value_by_key[k] for k in keys], use_std)
            for j, key in enumerate(keys):
                normalized_by_key[key] = float(vals[j])
        return [normalized_by_key[key] for key in key_by_sample]

    group_to_indices: dict[int, list[int]] = {}
    for i, sample in enumerate(samples):
        group_to_indices.setdefault(_sample_group_key(sample), []).append(i)

    normalized = list(values)
    for idxs in group_to_indices.values():
        vals = _normalize_values([values[i] for i in idxs], use_std)
        for j, sample_idx in enumerate(idxs):
            normalized[sample_idx] = float(vals[j])
    return normalized


def _default_post_process(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    """Mirror slime's default reward post-process for GRPO/GSPO.

    This function is only used when EXPLORE_ADVANTAGE_BONUS is enabled; keeping
    the default math here lets us add post-normalization exploration bonuses
    without replacing the rest of slime's behavior.
    """
    raw_rewards = [_reward_value(args, sample) for sample in samples]
    if (
        getattr(args, "advantage_estimator", None) in ["grpo", "gspo"]
        and getattr(args, "rewards_normalization", False)
    ):
        return raw_rewards, _group_normalize_sample_values(args, samples, raw_rewards)

    return raw_rewards, raw_rewards


def _dual_stream_post_process(
    args: Any,
    samples: list[Any],
    base_rewards: list[float],
) -> list[float]:
    intrinsic_key = _env_str(
        "EXPLORE_ADVANTAGE_INTRINSIC_KEY",
        "explore_agent57_intrinsic_signal",
    )
    lambda_coef = _env_float(
        "EXPLORE_ADVANTAGE_LAMBDA",
        _env_float("EXPLORE_ADVANTAGE_BONUS_COEF", 0.1),
    )
    arm_weight_mode = _env_str("EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE", "normalized_beta").lower()
    trust_key = _env_str("EXPLORE_ADVANTAGE_TRUST_KEY", "explore_agent57_trust")
    clip = _env_float("EXPLORE_ADVANTAGE_BONUS_CLIP", 0.0)

    intrinsic_values = [_component_value(sample, intrinsic_key) for sample in samples]
    intrinsic_adv = _group_normalize_sample_values(args, samples, intrinsic_values)

    betas = [_component_value(sample, "explore_agent57_beta") for sample in samples]
    max_beta = max([abs(beta) for beta in betas if beta > 0.0] or [1.0])
    adjusted = list(base_rewards)
    for i, sample in enumerate(samples):
        if arm_weight_mode in {"none", "off", "0"}:
            arm_weight = 1.0
        elif arm_weight_mode in {"raw", "raw_beta"}:
            arm_weight = max(0.0, betas[i])
        else:
            arm_weight = max(0.0, betas[i]) / max(max_beta, 1e-12)
        reward = getattr(sample, "reward", None)
        trust_missing = not isinstance(reward, dict) or trust_key not in reward
        trust = _component_value(sample, trust_key)
        if trust_missing and trust_key == "explore_agent57_trust":
            trust = 1.0
        raw_bonus = float(lambda_coef * arm_weight * trust * intrinsic_adv[i])
        bonus = max(-clip, min(clip, raw_bonus)) if clip > 0 else raw_bonus
        adjusted[i] += bonus
        if isinstance(reward, dict):
            reward["explore_post_norm_base_reward"] = base_rewards[i]
            reward["explore_post_norm_intrinsic_value"] = intrinsic_values[i]
            reward["explore_post_norm_bonus_raw"] = raw_bonus
            reward["explore_post_norm_bonus"] = bonus
            reward["explore_post_norm_bonus_coef"] = lambda_coef
            reward["explore_post_norm_bonus_clip"] = clip
            reward["explore_post_norm_bonus_mode"] = "dual_stream"
            reward["explore_post_norm_intrinsic_key"] = intrinsic_key
            reward["explore_post_norm_intrinsic_advantage"] = intrinsic_adv[i]
            reward["explore_post_norm_arm_weight"] = arm_weight
            reward["explore_post_norm_trust"] = trust
            reward["explore_post_norm_adjusted_reward"] = adjusted[i]
            reward["postprocess_total_reward"] = adjusted[i]
            _sync_reward_aliases(
                reward,
                total_reward=adjusted[i],
                extra_exploration_reward=bonus,
            )
    return adjusted


def post_process_rewards(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    raw_rewards, rewards = _default_post_process(args, samples)
    if not _env_flag("EXPLORE_ADVANTAGE_BONUS_ENABLED", os.getenv("EXPLORE_ADVANTAGE_BONUS", "0")):
        return raw_rewards, rewards
    mode = _env_str("EXPLORE_ADVANTAGE_BONUS_MODE", "component").lower()
    if mode in {"dual", "dual_stream", "intrinsic_advantage"}:
        return raw_rewards, _dual_stream_post_process(args, samples, rewards)

    component_names = [
        part.strip()
        for part in os.getenv("EXPLORE_ADVANTAGE_BONUS_COMPONENTS", "explore_intrinsic_scaled").split(",")
        if part.strip()
    ]
    coef = _env_float("EXPLORE_ADVANTAGE_BONUS_COEF", 1.0)
    clip = _env_float("EXPLORE_ADVANTAGE_BONUS_CLIP", 0.25)

    adjusted = list(rewards)
    for i, sample in enumerate(samples):
        raw_bonus = sum(_component_value(sample, key) for key in component_names)
        clipped_bonus = max(-clip, min(clip, raw_bonus)) if clip > 0 else raw_bonus
        bonus = coef * clipped_bonus
        adjusted[i] += bonus
        reward = getattr(sample, "reward", None)
        if isinstance(reward, dict):
            reward["explore_post_norm_base_reward"] = rewards[i]
            reward["explore_post_norm_bonus_raw"] = raw_bonus
            reward["explore_post_norm_bonus"] = bonus
            reward["explore_post_norm_bonus_coef"] = coef
            reward["explore_post_norm_bonus_clip"] = clip
            reward["explore_post_norm_bonus_mode"] = "component"
            reward["explore_post_norm_bonus_components"] = ",".join(component_names)
            reward["explore_post_norm_adjusted_reward"] = adjusted[i]
            reward["postprocess_total_reward"] = adjusted[i]
            _sync_reward_aliases(
                reward,
                total_reward=adjusted[i],
                extra_exploration_reward=bonus,
            )
    return raw_rewards, adjusted
