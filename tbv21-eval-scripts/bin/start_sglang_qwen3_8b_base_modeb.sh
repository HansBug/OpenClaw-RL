#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env.sh"

mkdir -p "${TBV21_LOG_DIR}/sglang" "${TBV21_STATE_DIR}"

GPU_IDS="${1:-${TBV21_GPU_IDS:-0,1,2,3}}"
TP_SIZE="${TP_SIZE:-$(awk -F, '{print NF}' <<< "${GPU_IDS}")}"
HOST="${TBV21_SGLANG_HOST:-127.0.0.1}"
PORT="${TBV21_SGLANG_PORT:-30000}"
MODEL_PATH="${TBV21_MODEL_PATH:?TBV21_MODEL_PATH is required}"
MODEL_NAME="${TBV21_MODEL_NAME:-qwen3-8b-base}"
LOG_FILE="${TBV21_LOG_DIR}/sglang/sglang_${MODEL_NAME}_${PORT}_modeb.log"
PID_FILE="${TBV21_STATE_DIR}/sglang_${PORT}.pid"
TMUX_SESSION="${TBV21_SGLANG_TMUX_SESSION:-tbv21_sglang_${PORT}}"

if curl --noproxy '*' -fsS --max-time 5 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[OK] sglang already ready port=${PORT}"
  curl --noproxy '*' -fsS --max-time 5 "http://${HOST}:${PORT}/get_server_info" || true
  echo
  exit 0
fi

if [ ! -d "${MODEL_PATH}" ]; then
  echo "[ERROR] model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

CMD=(
  "${TBV21_SGLANG_PYTHON}" -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --tokenizer-path "${MODEL_PATH}"
  --served-model-name "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tp-size "${TP_SIZE}"
  --base-gpu-id 0
  --dp-size 1
  --mem-fraction-static "${TBV21_SGLANG_MEM_FRACTION:-0.6}"
  --attention-backend fa3
  --sampling-backend flashinfer
  --grammar-backend xgrammar
  --chunked-prefill-size 8192
  --max-prefill-tokens 16384
  --schedule-policy fcfs
  --page-size 1
  --dtype auto
  --kv-cache-dtype auto
  --context-length 40960
  --random-seed 1234
  --trust-remote-code
)

printf '%q ' "${CMD[@]}" > "${TBV21_STATE_DIR}/sglang_${PORT}.cmd"
echo >> "${TBV21_STATE_DIR}/sglang_${PORT}.cmd"

tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" \
  "cd '${TBV21_HOME}' && source '${TBV21_HOME}/env.sh' && export CUDA_VISIBLE_DEVICES='${GPU_IDS}' PYTHONUNBUFFERED=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 && exec bash '${TBV21_STATE_DIR}/sglang_${PORT}.cmd' >> '${LOG_FILE}' 2>&1"

echo "[INFO] starting sglang session=${TMUX_SESSION}"
echo "[INFO] model=${MODEL_PATH}"
echo "[INFO] served=${MODEL_NAME} gpu_ids=${GPU_IDS} tp=${TP_SIZE} port=${PORT}"
echo "[INFO] log=${LOG_FILE}"

for i in $(seq 1 900); do
  if curl --noproxy '*' -fsS --max-time 5 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[OK] sglang ready after ${i}s"
    curl --noproxy '*' -fsS --max-time 5 "http://${HOST}:${PORT}/get_server_info" || true
    echo
    exit 0
  fi
  if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "[ERROR] sglang tmux session exited early"
    tail -n 240 "${LOG_FILE}" || true
    exit 1
  fi
  sleep 1
done

echo "[ERROR] sglang not ready after timeout"
tail -n 240 "${LOG_FILE}" || true
exit 1
