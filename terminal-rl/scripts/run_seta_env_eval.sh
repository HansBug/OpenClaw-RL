#!/usr/bin/env bash
# Run one SETA-env accuracy eval pass over a JSONL, with no training.
#
# This drives slime/eval_only.py through the existing SETA DAPO launcher, which
# is why the wrapper carries a `dapo` name: with SLIME_ENTRYPOINT pointed at
# eval_only.py there is no training loss and the checkpoint is loaded read-only.
#
# A full 1356-sample pass usually leaves some samples without a result, because
# remote Docker resets fail for a few tasks. The intended loop is:
#   1. run this script over the full dataset
#   2. analyze, writing a supplement JSONL of the samples with no result
#   3. run this script again over that supplement, at lower concurrency
#   4. analyze all passes together; later passes win
# See docs/SETA_ENV_EVAL_zh.md for the worked example.
#
# Required:
#   HF_CKPT       checkpoint to evaluate; also used as REF_LOAD unless set
#   WORKER_URLS   Docker worker endpoint(s) for the terminal environment
#   ENV_SERVER_URL  environment router endpoint
#
# Optional:
#   PROMPT_DATA=terminal-rl/dataset/seta_env_convert/train.filtered.jsonl
#   CONCURRENCY=16    drives batch size, eval concurrency and active task cap
#   RUN_ID            defaults to the launcher's own generated id
#   DRY_RUN=1         print the resolved environment and exit
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TERMINAL_RL_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${TERMINAL_RL_DIR}/.." &>/dev/null && pwd)"

: "${HF_CKPT:?HF_CKPT is required (checkpoint to evaluate)}"
: "${WORKER_URLS:?WORKER_URLS is required (Docker worker endpoints)}"
: "${ENV_SERVER_URL:?ENV_SERVER_URL is required (environment router endpoint)}"

export HF_CKPT
export REF_LOAD="${REF_LOAD:-${HF_CKPT}}"
export WORKER_URLS ENV_SERVER_URL

PROMPT_DATA="${PROMPT_DATA:-${TERMINAL_RL_DIR}/dataset/seta_env_convert/train.filtered.jsonl}"
if [[ ! -r "${PROMPT_DATA}" ]]; then
  echo "[ERROR] PROMPT_DATA ${PROMPT_DATA} is missing or unreadable" >&2
  exit 1
fi
export ROLLOUT_PROMPT_DATA="${PROMPT_DATA}"
export EVAL_PROMPT_DATA="${PROMPT_DATA}"

# eval_only.py is what makes this a read-only accuracy pass rather than training.
export SLIME_ENTRYPOINT="${SLIME_ENTRYPOINT:-${REPO_ROOT}/slime/eval_only.py}"
if [[ ! -r "${SLIME_ENTRYPOINT}" ]]; then
  echo "[ERROR] SLIME_ENTRYPOINT ${SLIME_ENTRYPOINT} is missing or unreadable" >&2
  exit 1
fi

# One knob for the three settings that must move together; raising any of them
# alone just shifts where the queue backs up. Lower it for supplement passes:
# the published run recovered 5 of its last 7 samples by dropping 16 to 2.
CONCURRENCY="${CONCURRENCY:-16}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-${CONCURRENCY}}"
export EVAL_ROLLOUT_MAX_CONCURRENCY="${EVAL_ROLLOUT_MAX_CONCURRENCY:-${CONCURRENCY}}"
export ENV_REMOTE_MAX_ACTIVE_TASKS="${ENV_REMOTE_MAX_ACTIVE_TASKS:-${CONCURRENCY}}"

# Nothing is trained, so keeping checkpoints only risks filling a directory the
# evaluating user may not even be able to write to.
export MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-0}"

echo "[INFO] checkpoint    ${HF_CKPT}"
echo "[INFO] prompt data   ${ROLLOUT_PROMPT_DATA} ($(wc -l < "${ROLLOUT_PROMPT_DATA}") samples)"
echo "[INFO] entrypoint    ${SLIME_ENTRYPOINT}"
echo "[INFO] concurrency   ${CONCURRENCY}"
echo "[INFO] env server    ${ENV_SERVER_URL}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[INFO] DRY_RUN=1, not launching"
  exit 0
fi

exec bash "${TERMINAL_RL_DIR}/terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh" "$@"
