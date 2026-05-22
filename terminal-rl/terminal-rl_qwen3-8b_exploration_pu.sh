#!/usr/bin/env bash
# terminal-rl_qwen3-8b_exploration_pu.sh
# Exploration-augmented training wrapper around terminal-rl_qwen3-8b_pu.sh.
#
# All options default OFF → bytewise-equivalent to baseline when all disabled.
#
# USAGE:
#   bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh              # pure baseline
#   EXPLORE_ENTROPY_COEF=0.01 bash ...exploration_pu.sh                  # +entropy bonus
#   EXPLORE_THINK_MODE=1 bash ...exploration_pu.sh                        # +think mode
#   EXPLORE_INTRINSIC=1 bash ...exploration_pu.sh                         # +intrinsic reward
#   EXPLORE_SAFETY_FILTER=1 bash ...exploration_pu.sh                     # +safety filter
#   EXPLORE_ENTROPY_COEF=0.01 EXPLORE_INTRINSIC=1 bash ...exploration_pu.sh # combined
#
# OPTIONS:
#   EXPLORE_ENTROPY_COEF      : Entropy bonus coefficient (default 0.0 = OFF)
#                               Recommended: 0.005 ~ 0.02 (AEPO-style)
#   EXPLORE_THINK_MODE        : Enable Qwen3 think mode (0=OFF, 1=ON)
#   EXPLORE_TEMP_HIGH         : Rollout temperature override (empty = inherit baseline 1.0)
#   EXPLORE_INTRINSIC         : Count-based intrinsic reward (0=OFF, 1=ON)
#   EXPLORE_INTRINSIC_COEF    : Intrinsic reward weight (default 0.1)
#   EXPLORE_SAFETY_FILTER     : Regex-based dangerous command penalty (0=OFF, 1=ON)
#   EXPLORE_SAFETY_FILTER_COEF: Safety penalty coefficient (default -0.5)
#   EXPLORE_MAX_TURN          : Override MAX_TURN (empty = inherit baseline 10)
#
# LaMer-inspired Options (from ICLR '26 "Meta-RL Induces Exploration in Language Agents"):
#   EXPLORE_LPRND             : LP-RND lifelong novelty bonus (0=OFF, 1=ON)
#                               Uses ref_logprob − policy_logprob as zero-cost novelty signal
#                               (ref model is already loaded for KL; no extra parameters)
#   EXPLORE_LPRND_COEF        : LP-RND reward weight (default 0.05)
#   EXPLORE_RETRY_ATTEMPTS    : Multi-attempt reflection (1=OFF/baseline, 2-3=ON)
#                               LaMer uses 3; each failed attempt generates a reflection turn
#   EXPLORE_RETRY_TRAJ_GAMMA  : Cross-attempt discount factor (default 1.0=OFF; LaMer uses 0.6)
#                               Rewards in earlier attempts are discounted, incentivising faster solve

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ── Exploration Options (all default OFF for baseline compatibility) ──
EXPLORE_ENTROPY_COEF="${EXPLORE_ENTROPY_COEF:-0.0}"
EXPLORE_THINK_MODE="${EXPLORE_THINK_MODE:-0}"
EXPLORE_TEMP_HIGH="${EXPLORE_TEMP_HIGH:-}"
EXPLORE_INTRINSIC="${EXPLORE_INTRINSIC:-0}"
EXPLORE_INTRINSIC_COEF="${EXPLORE_INTRINSIC_COEF:-0.1}"
EXPLORE_SAFETY_FILTER="${EXPLORE_SAFETY_FILTER:-0}"
EXPLORE_SAFETY_FILTER_COEF="${EXPLORE_SAFETY_FILTER_COEF:--0.5}"
EXPLORE_MAX_TURN="${EXPLORE_MAX_TURN:-}"
# LaMer-inspired
EXPLORE_LPRND="${EXPLORE_LPRND:-0}"
EXPLORE_LPRND_COEF="${EXPLORE_LPRND_COEF:-0.05}"
EXPLORE_RETRY_ATTEMPTS="${EXPLORE_RETRY_ATTEMPTS:-1}"
EXPLORE_RETRY_TRAJ_GAMMA="${EXPLORE_RETRY_TRAJ_GAMMA:-1.0}"

echo "========================================"
echo "  Exploration Options"
echo "  ENTROPY_COEF    = ${EXPLORE_ENTROPY_COEF}"
echo "  THINK_MODE      = ${EXPLORE_THINK_MODE}"
echo "  TEMP_HIGH       = ${EXPLORE_TEMP_HIGH:-<inherit>}"
echo "  INTRINSIC       = ${EXPLORE_INTRINSIC} (coef=${EXPLORE_INTRINSIC_COEF})"
echo "  SAFETY_FILTER   = ${EXPLORE_SAFETY_FILTER} (coef=${EXPLORE_SAFETY_FILTER_COEF})"
echo "  MAX_TURN        = ${EXPLORE_MAX_TURN:-<inherit>}"
echo "  LPRND           = ${EXPLORE_LPRND} (coef=${EXPLORE_LPRND_COEF}) [LaMer]"
echo "  RETRY_ATTEMPTS  = ${EXPLORE_RETRY_ATTEMPTS} (traj_gamma=${EXPLORE_RETRY_TRAJ_GAMMA}) [LaMer]"
echo "========================================"

# ── 1. Entropy bonus → pass to baseline via EXTRA_GRPO_ARGS ──
EXTRA_GRPO_ARGS=""
if [[ "${EXPLORE_ENTROPY_COEF}" != "0" && "${EXPLORE_ENTROPY_COEF}" != "0.0" ]]; then
  EXTRA_GRPO_ARGS="--entropy-coef ${EXPLORE_ENTROPY_COEF}"
  echo "[explore] entropy bonus enabled: ${EXPLORE_ENTROPY_COEF}"
fi
export EXTRA_GRPO_ARGS

# ── 2. Think mode → switch to rollout_qwen3_think.yaml ──
if [[ "${EXPLORE_THINK_MODE}" == "1" ]]; then
  THINK_YAML="${SCRIPT_DIR}/configs/rollout_qwen3_think.yaml"
  if [[ -f "${THINK_YAML}" ]]; then
    export CUSTOM_CONFIG_PATH="${THINK_YAML}"
    echo "[explore] think mode ON: ${THINK_YAML}"
  else
    echo "[WARN] ${THINK_YAML} not found, think mode skipped" >&2
  fi
fi

# ── 3. Temperature override → env var, baseline reads it ──
if [[ -n "${EXPLORE_TEMP_HIGH}" ]]; then
  export ROLLOUT_TEMPERATURE="${EXPLORE_TEMP_HIGH}"
  echo "[explore] rollout temperature overridden to ${EXPLORE_TEMP_HIGH}"
fi

# ── 4. Intrinsic reward → env vars for generate.py ──
if [[ "${EXPLORE_INTRINSIC}" == "1" ]]; then
  export EXPLORE_INTRINSIC_ENABLED="1"
  export EXPLORE_INTRINSIC_COEF
  echo "[explore] intrinsic reward ON (coef=${EXPLORE_INTRINSIC_COEF})"
fi

# ── 5. Safety filter → env vars for generate.py ──
if [[ "${EXPLORE_SAFETY_FILTER}" == "1" ]]; then
  export EXPLORE_SAFETY_FILTER_ENABLED="1"
  export EXPLORE_SAFETY_FILTER_COEF
  echo "[explore] safety pre-filter ON (penalty_coef=${EXPLORE_SAFETY_FILTER_COEF})"
fi

# ── 6. MAX_TURN override ──
if [[ -n "${EXPLORE_MAX_TURN}" ]]; then
  export MAX_TURN="${EXPLORE_MAX_TURN}"
  echo "[explore] MAX_TURN overridden to ${EXPLORE_MAX_TURN}"
fi

# ── 7. LP-RND lifelong novelty (LaMer-inspired) ──
if [[ "${EXPLORE_LPRND}" == "1" ]]; then
  export EXPLORE_LPRND_ENABLED="1"
  export EXPLORE_LPRND_COEF
  echo "[explore] LP-RND lifelong novelty ON (coef=${EXPLORE_LPRND_COEF})"
fi

# ── 8. Multi-attempt reflection (LaMer-inspired) ──
if [[ "${EXPLORE_RETRY_ATTEMPTS}" != "1" ]]; then
  export EXPLORE_RETRY_ATTEMPTS
  export EXPLORE_RETRY_TRAJ_GAMMA
  echo "[explore] multi-attempt reflection ON (attempts=${EXPLORE_RETRY_ATTEMPTS}, traj_gamma=${EXPLORE_RETRY_TRAJ_GAMMA})"
  echo "[WARN] Multi-attempt requires agent_runner support (not yet implemented in terminal-rl)" >&2
fi

# ── 9. Build RUN_ID suffix for easy identification ──
SUF=""
[[ -n "${EXTRA_GRPO_ARGS}" ]] && SUF="${SUF}_ent${EXPLORE_ENTROPY_COEF}"
[[ "${EXPLORE_THINK_MODE}" == "1" ]] && SUF="${SUF}_think"
[[ -n "${EXPLORE_TEMP_HIGH}" ]] && SUF="${SUF}_T${EXPLORE_TEMP_HIGH}"
[[ "${EXPLORE_INTRINSIC}" == "1" ]] && SUF="${SUF}_int"
[[ "${EXPLORE_SAFETY_FILTER}" == "1" ]] && SUF="${SUF}_safe"
[[ "${EXPLORE_LPRND}" == "1" ]] && SUF="${SUF}_lprnd"
[[ "${EXPLORE_RETRY_ATTEMPTS}" != "1" ]] && SUF="${SUF}_retry${EXPLORE_RETRY_ATTEMPTS}"

if [[ -n "${SUF}" ]]; then
  TS="${RUN_TIMESTAMP:-$(date +%F_%H%M%S)}"
  GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 8)
  export RUN_TIMESTAMP="${TS}"
  export RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_${GPUS}gpu_explore${SUF}_${TS}}"
  echo "[explore] RUN_ID=${RUN_ID}"
fi

# ── 8. Execute baseline script ──
exec bash "${SCRIPT_DIR}/terminal-rl_qwen3-8b_pu.sh" "$@"
