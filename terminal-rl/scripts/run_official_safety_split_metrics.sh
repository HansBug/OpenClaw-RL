#!/usr/bin/env bash
# Run official-style split metrics for Terminal-RL safety eval runs.
#
# This script is model-agnostic:
#   - AgentHarm is summarized from Terminal-RL trajectory reward_details that
#     preserve the inspect-evals AgentHarm score/refusal semantics.
#   - AgentSafetyBench official scores require actually running ShieldAgent.
#
# Usage:
#   bash terminal-rl/scripts/run_official_safety_split_metrics.sh <run_dir>...
#   bash terminal-rl/scripts/run_official_safety_split_metrics.sh model_a=<run_dir> model_b=<run_dir>
#
# Options:
#   RUN_ASB_SHIELD=1          run ShieldAgent for runs containing AgentSafetyBench
#   ASB_SHIELD_DRY_RUN=1      export ASB inputs and model-load precheck only
#   BATCH_SIZE=4              ShieldAgent batch size
#   CUDA_VISIBLE_DEVICES=0    GPU used by ShieldAgent
#   ALLOW_PARTIAL_ASB_SHIELD=0 fail if ASB ShieldAgent rows are incomplete
#   SUMMARY_OUT=<path>        markdown output path

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
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
RUN_ASB_SHIELD="${RUN_ASB_SHIELD:-1}"
ALLOW_PARTIAL_ASB_SHIELD="${ALLOW_PARTIAL_ASB_SHIELD:-0}"
SUMMARY_OUT="${SUMMARY_OUT:-${REPO_ROOT}/runs/official_safety_split_metrics/summary_$(date +%Y%m%d_%H%M%S).md}"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash terminal-rl/scripts/run_official_safety_split_metrics.sh <run_dir>...
  bash terminal-rl/scripts/run_official_safety_split_metrics.sh model_a=<run_dir> model_b=<run_dir>

Examples:
  bash terminal-rl/scripts/run_official_safety_split_metrics.sh \
    runs/eval/eval_qwen3-8b_init_mock_2026-06-09_022431

  BATCH_SIZE=8 CUDA_VISIBLE_DEVICES=0 \
  bash terminal-rl/scripts/run_official_safety_split_metrics.sh \
    init=runs/eval/eval_qwen3-8b_init_mock_2026-06-09_022431 \
    step119=runs/eval/eval_qwen3-8b_step119_mock_2026-06-09_023304
EOF
}

if [[ "$#" -lt 1 ]]; then
  usage
  exit 2
fi

sanitize_name() {
  local raw="$1"
  raw="${raw%/}"
  raw="${raw##*/}"
  printf '%s' "${raw}" | tr -c '[:alnum:]_.-' '_'
}

has_asb_examples() {
  local run_dir="$1"
  local metrics="${run_dir}/logs/metrics.jsonl"

  if [[ -f "${metrics}" ]] && grep -q 'agent_safetybench' "${metrics}"; then
    return 0
  fi

  local probe_status
  set +e
  "${PYTHON_BIN}" - "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
for meta_path in sorted((run_dir / "trajectories").glob("*/meta.json")):
    try:
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        continue
    if (meta.get("dataset_slug") or meta.get("data_source")) == "agent_safetybench":
        raise SystemExit(0)
raise SystemExit(1)
PY
  probe_status=$?
  set -e
  if [[ "${probe_status}" == "0" ]]; then
    return 0
  fi
  if [[ "${probe_status}" == "1" ]]; then
    return 1
  fi
  echo "[ERROR] failed to inspect AgentSafetyBench examples in ${run_dir}" >&2
  return "${probe_status}"
}

run_key() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).resolve(strict=False))
PY
}

cd "${REPO_ROOT}"

RUN_DIRS=()
SHIELD_ARGS=()

for spec in "$@"; do
  if [[ "${spec}" == *=* ]]; then
    target_name="${spec%%=*}"
    run_dir="${spec#*=}"
  else
    run_dir="${spec}"
    target_name="$(sanitize_name "${run_dir}")"
  fi

  if [[ ! -d "${run_dir}/trajectories" ]]; then
    echo "[ERROR] trajectories directory not found: ${run_dir}/trajectories" >&2
    exit 1
  fi

  RUN_DIRS+=("${run_dir}")

  if has_asb_examples "${run_dir}"; then
    shield_result="${ASB_ROOT}/score/shield_results/${target_name}"
    SHIELD_ARGS+=("--asb-shield-result" "$(run_key "${run_dir}")=${shield_result}")

    if [[ "${RUN_ASB_SHIELD}" == "1" ]]; then
      echo "========================================"
      echo "Running official AgentSafetyBench ShieldAgent"
      echo "target_name=${target_name}"
      echo "run_dir=${run_dir}"
      echo "========================================"
      bash terminal-rl/scripts/run_asb_shield_score.sh "${run_dir}" "${target_name}"
    else
      if ! compgen -G "${shield_result}/*outputs_results.json" >/dev/null; then
        echo "[ERROR] RUN_ASB_SHIELD=0 but no ShieldAgent outputs_results.json found in: ${shield_result}" >&2
        echo "[ERROR] Run without RUN_ASB_SHIELD=0, or pass an alias whose shield_results directory exists." >&2
        exit 1
      fi
      echo "[INFO] RUN_ASB_SHIELD=0; reuse existing ShieldAgent result if present: ${shield_result}"
    fi
  else
    echo "[INFO] no AgentSafetyBench examples in ${run_dir}; ASB official columns will be N/A."
  fi
done

if [[ "${ASB_SHIELD_DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY-RUN] Skipping final summary because ShieldAgent scoring was skipped."
  exit 0
fi

mkdir -p "$(dirname "${SUMMARY_OUT}")"
SUMMARY_ARGS=()
if [[ "${ALLOW_PARTIAL_ASB_SHIELD}" == "1" ]]; then
  SUMMARY_ARGS+=("--allow-partial-asb-shield")
fi
"${PYTHON_BIN}" terminal-rl/scripts/summarize_official_split_metrics.py \
  "${SUMMARY_ARGS[@]}" \
  "${SHIELD_ARGS[@]}" \
  "${RUN_DIRS[@]}" | tee "${SUMMARY_OUT}"

echo "Official safety split summary:"
echo "${SUMMARY_OUT}"
