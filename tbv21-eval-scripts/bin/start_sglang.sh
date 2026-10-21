#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env.sh"

mkdir -p "${TBV21_LOG_DIR}/sglang" "${TBV21_STATE_DIR}"

GPU_IDS="${1:-${TBV21_GPU_IDS:-0,1,2,3}}"
TP_SIZE="${TP_SIZE:-$(awk -F, '{print NF}' <<< "${GPU_IDS}")}"
HOST="${TBV21_SGLANG_HOST}"
PORT="${TBV21_SGLANG_PORT}"
LOG_FILE="${TBV21_LOG_DIR}/sglang/sglang_${TBV21_MODEL_NAME}_${PORT}.log"
PID_FILE="${TBV21_STATE_DIR}/sglang_${PORT}.pid"
TMUX_SESSION="${TBV21_SGLANG_TMUX_SESSION:-tbv21_sglang_${PORT}}"

if [ -f "${PID_FILE}" ]; then
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [ -n "${pid}" ] && ps -p "${pid}" >/dev/null 2>&1; then
    if curl --noproxy '*' -fsS --max-time 5 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[OK] sglang already ready pid=${pid} port=${PORT}"
      exit 0
    fi
    echo "[WARN] stale sglang pid=${pid}; stopping"
    kill "${pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${pid}" 2>/dev/null || true
  fi
fi

if curl --noproxy '*' -fsS --max-time 5 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[OK] sglang already ready port=${PORT}"
  curl --noproxy '*' -fsS --max-time 5 "http://${HOST}:${PORT}/v1/models" || true
  echo
  exit 0
fi

if [ ! -d "${TBV21_MODEL_PATH}" ]; then
  echo "[ERROR] model path not found: ${TBV21_MODEL_PATH}" >&2
  exit 1
fi

echo "[INFO] starting sglang"
echo "[INFO] model=${TBV21_MODEL_PATH}"
echo "[INFO] served=${TBV21_MODEL_NAME}"
echo "[INFO] gpu_ids=${GPU_IDS} tp=${TP_SIZE} port=${PORT}"
echo "[INFO] log=${LOG_FILE}"

CMD=(
  "${TBV21_SGLANG_PYTHON}" -m sglang.launch_server
  --model-path "${TBV21_MODEL_PATH}"
  --served-model-name "${TBV21_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tp-size "${TP_SIZE}"
  --trust-remote-code
  --mem-fraction-static "${TBV21_SGLANG_MEM_FRACTION}"
)
if [ "${TBV21_SGLANG_DISABLE_CUDA_GRAPH}" = "1" ]; then
  CMD+=(--disable-cuda-graph)
fi

printf '%q ' "${CMD[@]}" > "${TBV21_STATE_DIR}/sglang_${PORT}.cmd"
echo >> "${TBV21_STATE_DIR}/sglang_${PORT}.cmd"

if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${TMUX_SESSION}" \
    "cd '${TBV21_HOME}' && source '${TBV21_HOME}/env.sh' && export CUDA_VISIBLE_DEVICES='${GPU_IDS}' PYTHONUNBUFFERED=1 && exec bash '${TBV21_STATE_DIR}/sglang_${PORT}.cmd' >> '${LOG_FILE}' 2>&1"
  echo "[INFO] tmux session=${TMUX_SESSION}"
else
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" PYTHONUNBUFFERED=1 setsid "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
fi

for i in $(seq 1 900); do
  if curl --noproxy '*' -fsS --max-time 5 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[OK] sglang ready after ${i}s"
    curl --noproxy '*' -fsS --max-time 5 "http://${HOST}:${PORT}/v1/models" || true
    echo
    exit 0
  fi
  if command -v tmux >/dev/null 2>&1; then
    if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
      echo "[ERROR] sglang tmux session exited early"
      tail -n 200 "${LOG_FILE}" || true
      exit 1
    fi
  else
    if ! ps -p "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
      echo "[ERROR] sglang exited early"
      tail -n 200 "${LOG_FILE}" || true
      exit 1
    fi
  fi
  sleep 1
done

echo "[ERROR] sglang not ready after timeout"
tail -n 240 "${LOG_FILE}" || true
exit 1
