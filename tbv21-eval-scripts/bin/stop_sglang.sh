#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env.sh"

PID_FILE="${TBV21_STATE_DIR}/sglang_${TBV21_SGLANG_PORT}.pid"
TMUX_SESSION="${TBV21_SGLANG_TMUX_SESSION:-tbv21_sglang_${TBV21_SGLANG_PORT}}"
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "[INFO] stopping tmux session ${TMUX_SESSION}"
  tmux kill-session -t "${TMUX_SESSION}" || true
fi
if [ -f "${PID_FILE}" ]; then
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [ -n "${pid}" ] && ps -p "${pid}" >/dev/null 2>&1; then
    echo "[INFO] stopping sglang pid=${pid}"
    kill "${pid}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      ps -p "${pid}" >/dev/null 2>&1 || break
      sleep 1
    done
    kill -9 "${pid}" 2>/dev/null || true
  fi
fi
rm -f "${PID_FILE}"
echo "[OK] sglang stopped"
