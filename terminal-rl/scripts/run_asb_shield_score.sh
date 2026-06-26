#!/usr/bin/env bash
# Export Terminal-RL AgentSafetyBench trajectories and score them with the
# official Agent-SafetyBench ShieldAgent scorer.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

RUN_DIR="${1:?Usage: bash terminal-rl/scripts/run_asb_shield_score.sh <run_dir> [target_name]}"
TARGET_NAME="${2:-$(basename "${RUN_DIR}")}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ASB_ROOT="${ASB_ROOT:-${AGENT_SAFETYBENCH_ROOT:-}}"
if [[ -z "${ASB_ROOT}" ]]; then
  for candidate in \
    "${REPO_ROOT}/../Agent-SafetyBench" \
    "${REPO_ROOT}/external/Agent-SafetyBench"; do
    if [[ -d "${candidate}/score" ]]; then
      ASB_ROOT="${candidate}"
      break
    fi
  done
fi
ASB_ROOT="${ASB_ROOT:-${REPO_ROOT}/../Agent-SafetyBench}"

DEFAULT_SHIELD_MODEL_ALIAS="${REPO_ROOT}/runs/models/ShieldAgent"
DEFAULT_SHIELD_MODEL_SOURCE="${SHIELD_MODEL_SOURCE:-}"
if [[ -z "${DEFAULT_SHIELD_MODEL_SOURCE}" ]]; then
  for candidate in \
    "${DEFAULT_SHIELD_MODEL_ALIAS}"; do
    if [[ -d "${candidate}" ]]; then
      DEFAULT_SHIELD_MODEL_SOURCE="${candidate}"
      break
    fi
  done
fi
DEFAULT_SHIELD_MODEL_SOURCE="${DEFAULT_SHIELD_MODEL_SOURCE:-${DEFAULT_SHIELD_MODEL_ALIAS}}"

if [[ -n "${SHIELD_MODEL:-}" ]]; then
  SHIELD_MODEL_SOURCE="${SHIELD_MODEL}"
elif [[ -f "${DEFAULT_SHIELD_MODEL_ALIAS}/config.json" && -f "${DEFAULT_SHIELD_MODEL_ALIAS}/tokenizer_config.json" ]]; then
  SHIELD_MODEL_SOURCE="${DEFAULT_SHIELD_MODEL_ALIAS}"
else
  SHIELD_MODEL_SOURCE="${DEFAULT_SHIELD_MODEL_SOURCE}"
fi
SHIELD_MODEL_ALIAS="${SHIELD_MODEL_ALIAS:-${DEFAULT_SHIELD_MODEL_ALIAS}}"
if [[ "${SHIELD_MODEL_ALIAS}" == "${SHIELD_MODEL_SOURCE}" || "${SHIELD_MODEL_ALIAS}" == *"models--"* ]]; then
  echo "[WARN] SHIELD_MODEL_ALIAS points to cache/source path; reset to ${DEFAULT_SHIELD_MODEL_ALIAS}" >&2
  SHIELD_MODEL_ALIAS="${DEFAULT_SHIELD_MODEL_ALIAS}"
fi
BATCH_SIZE="${BATCH_SIZE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SHIELD_PRECHECK_TIMEOUT="${SHIELD_PRECHECK_TIMEOUT:-600}"
SHIELD_PRECHECK="${SHIELD_PRECHECK:-0}"
FORCE_ASB_EXPORT="${FORCE_ASB_EXPORT:-1}"
REUSE_ASB_SHIELD_RESULTS="${REUSE_ASB_SHIELD_RESULTS:-0}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/official_asb_shield_inputs/${TARGET_NAME}}"
LOG_DIR="${ASB_SHIELD_LOG_DIR:-${REPO_ROOT}/runs/official_asb_shield_logs/${TARGET_NAME}}"
LOG_FILE="${ASB_SHIELD_LOG_FILE:-${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================"
echo "AgentSafetyBench official ShieldAgent score"
echo "script_version=official_split_v1"
echo "run_dir=${RUN_DIR}"
echo "target_name=${TARGET_NAME}"
echo "log_file=${LOG_FILE}"
echo "========================================"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

run_maybe_timeout() {
  local seconds="$1"
  shift
  if [[ "${seconds}" != "0" ]] && command -v timeout >/dev/null 2>&1; then
    timeout "${seconds}" "$@"
  else
    "$@"
  fi
}

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

required = {
    "torch": "torch",
    "transformers": "transformers",
    "tqdm": "tqdm",
    "tabulate": "tabulate",
    "sklearn": "scikit-learn",
}
missing = [pkg for module, pkg in required.items() if importlib.util.find_spec(module) is None]
if missing:
    raise SystemExit(
        f"[ERROR] PYTHON_BIN is missing packages {missing}. "
        "Set PYTHON_BIN to the Agent-SafetyBench scoring environment."
    )
PY

if [[ ! -d "${ASB_ROOT}/score" ]]; then
  echo "[ERROR] Agent-SafetyBench score directory not found: ${ASB_ROOT}/score" >&2
  exit 1
fi

if [[ ! -d "${SHIELD_MODEL_SOURCE}" ]]; then
  echo "[ERROR] ShieldAgent source directory is not mounted or not a directory: ${SHIELD_MODEL_SOURCE}" >&2
  echo "[ERROR] Expected repo-local model at: ${DEFAULT_SHIELD_MODEL_ALIAS}" >&2
  echo "[ERROR] Or set SHIELD_MODEL to a local directory containing config.json/tokenizer_config.json." >&2
  exit 1
fi

if [[ ! -f "${SHIELD_MODEL_SOURCE}/config.json" || ! -f "${SHIELD_MODEL_SOURCE}/tokenizer_config.json" ]]; then
  SNAPSHOT_DIR="$(find "${SHIELD_MODEL_SOURCE}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1 || true)"
  if [[ -n "${SNAPSHOT_DIR}" && -f "${SNAPSHOT_DIR}/config.json" && -f "${SNAPSHOT_DIR}/tokenizer_config.json" ]]; then
    SHIELD_MODEL_SOURCE="${SNAPSHOT_DIR}"
  else
    echo "[ERROR] ShieldAgent local directory is missing config/tokenizer files: ${SHIELD_MODEL_SOURCE}" >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" - "${SHIELD_MODEL_SOURCE}" "${REPO_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
index_path = model_dir / "model.safetensors.index.json"
if not index_path.is_file():
    raise SystemExit(f"[ERROR] missing model.safetensors.index.json in {model_dir}")

with index_path.open(encoding="utf-8") as f:
    index = json.load(f)
shards = sorted(set(index.get("weight_map", {}).values()))
missing = [name for name in shards if not (model_dir / name).is_file()]
if missing:
    cmd = f"cd {repo_root} && bash terminal-rl/scripts/prepare_repo_local_shieldagent.sh"
    raise SystemExit(
        "[ERROR] ShieldAgent model shards are missing or broken in "
        f"{model_dir}: {missing}\n"
        "[ERROR] Prepare/download the repo-local model first:\n"
        f"  {cmd}\n"
        "[ERROR] If the source filesystem is unavailable, the script will download thu-coai/ShieldAgent."
    )
print(f"[OK] ShieldAgent shard check passed: {model_dir}; shards={len(shards)}")
PY

if [[ "${SHIELD_MODEL_SOURCE}" == "${DEFAULT_SHIELD_MODEL_ALIAS}" ]]; then
  SHIELD_MODEL="${SHIELD_MODEL_SOURCE}"
else
  mkdir -p "$(dirname "${SHIELD_MODEL_ALIAS}")"
  if [[ -L "${SHIELD_MODEL_ALIAS}" ]]; then
    rm -f "${SHIELD_MODEL_ALIAS}"
  fi
  mkdir -p "${SHIELD_MODEL_ALIAS}"
  find "${SHIELD_MODEL_ALIAS}" -mindepth 1 -maxdepth 1 -exec rm -rf '{}' +

  for item in "${SHIELD_MODEL_SOURCE}"/*; do
    name="$(basename "${item}")"
    case "${name}" in
      *.safetensors|*.bin)
        ln -sfn "${item}" "${SHIELD_MODEL_ALIAS}/${name}"
        ;;
      *)
        cp -aL "${item}" "${SHIELD_MODEL_ALIAS}/${name}"
        ;;
    esac
  done
  SHIELD_MODEL="${SHIELD_MODEL_ALIAS}"
fi

echo "shield_model_source=${SHIELD_MODEL_SOURCE}"
echo "shield_model_alias=${SHIELD_MODEL_ALIAS}"
echo "shield_model=${SHIELD_MODEL}"
echo "asb_root=${ASB_ROOT}"
echo "python_bin=${PYTHON_BIN}"
echo "batch_size=${BATCH_SIZE}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "force_asb_export=${FORCE_ASB_EXPORT}"
echo "reuse_asb_shield_results=${REUSE_ASB_SHIELD_RESULTS}"

if [[ "${SHIELD_PRECHECK}" == "1" ]]; then
  run_maybe_timeout "${SHIELD_PRECHECK_TIMEOUT}" "${PYTHON_BIN}" - "${SHIELD_MODEL}" <<'PY'
import sys
from transformers import AutoConfig, AutoTokenizer

model_path = sys.argv[1]
AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True, local_files_only=True)
print(f"[OK] ShieldAgent local load precheck passed: {model_path}")
PY
else
  echo "[INFO] skip AutoConfig/AutoTokenizer precheck. Set SHIELD_PRECHECK=1 to enable it."
fi

if [[ "${FORCE_ASB_EXPORT}" == "1" || ! -s "${OUT_DIR}/gen_res.json" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/export_asb_shield_inputs.py" "${RUN_DIR}" --out-dir "${OUT_DIR}" --filename gen_res.json
else
  echo "[INFO] Reusing existing exported ASB inputs: ${OUT_DIR}/gen_res.json"
  echo "[INFO] Set FORCE_ASB_EXPORT=1 to regenerate it."
fi

ASB_EXAMPLE_COUNT="$("${PYTHON_BIN}" - "${OUT_DIR}/gen_res.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="utf-8") as f:
    data = json.load(f)
print(len(data) if isinstance(data, list) else 0)
PY
)"
echo "asb_example_count=${ASB_EXAMPLE_COUNT}"
if [[ "${ASB_EXAMPLE_COUNT}" == "0" ]]; then
  echo "[ERROR] No AgentSafetyBench examples were exported from ${RUN_DIR}." >&2
  exit 1
fi

if [[ "${ASB_SHIELD_DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY-RUN] Skipping official ShieldAgent scoring."
  echo "[DRY-RUN] Exported inputs: ${OUT_DIR}/gen_res.json"
  exit 0
fi

SHIELD_RESULT_DIR="${ASB_ROOT}/score/shield_results/${TARGET_NAME}"
if [[ "${REUSE_ASB_SHIELD_RESULTS}" == "1" ]]; then
  echo "[INFO] Reusing existing ShieldAgent result dir if present: ${SHIELD_RESULT_DIR}"
else
  rm -rf "${SHIELD_RESULT_DIR}"
  echo "[INFO] Removed stale ShieldAgent result dir before scoring: ${SHIELD_RESULT_DIR}"
fi

cd "${ASB_ROOT}/score"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${PYTHON_BIN}" eval_with_shield.py \
  --model_path "${SHIELD_MODEL}" \
  --filepath "${OUT_DIR}" \
  --filename gen_res.json \
  --label_type "" \
  --batch_size "${BATCH_SIZE}" \
  --target_model_name "${TARGET_NAME}"

echo "Official ShieldAgent outputs:"
echo "${SHIELD_RESULT_DIR}/"
echo "Log file:"
echo "${LOG_FILE}"
