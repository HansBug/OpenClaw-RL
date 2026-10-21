#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env.sh"

if [ -f "${TBV21_DOCKER_PID}" ]; then
  pid="$(cat "${TBV21_DOCKER_PID}" 2>/dev/null || true)"
  if [ -n "${pid}" ] && ps -p "${pid}" -o comm= 2>/dev/null | grep -qx dockerd; then
    echo "[INFO] stopping tbv2.1 dockerd pid=${pid}"
    sudo kill "${pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      ps -p "${pid}" >/dev/null 2>&1 || break
      sleep 1
    done
    sudo kill -9 "${pid}" 2>/dev/null || true
  fi
fi
rm -f "${TBV21_DOCKER_SOCK}" "${TBV21_DOCKER_PID}"
echo "[OK] tbv2.1 Docker stopped"

