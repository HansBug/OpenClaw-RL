#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env.sh"

mkdir -p "${TBV21_DOCKER_DATA_ROOT}" "${TBV21_DOCKER_EXEC_ROOT}" "${TBV21_LOG_DIR}/docker" "${TBV21_STATE_DIR}"

if [ -S "${TBV21_DOCKER_SOCK}" ] && timeout 5 docker info >/dev/null 2>&1; then
  echo "[OK] tbv2.1 Docker already ready: ${DOCKER_HOST}"
  docker info | sed -n '1,80p'
  exit 0
fi

if [ -f "${TBV21_DOCKER_PID}" ]; then
  old_pid="$(cat "${TBV21_DOCKER_PID}" 2>/dev/null || true)"
  if [ -n "${old_pid}" ] && ps -p "${old_pid}" -o comm= 2>/dev/null | grep -qx dockerd; then
    echo "[WARN] stale tbv2.1 dockerd pid ${old_pid}; stopping"
    sudo kill "${old_pid}" 2>/dev/null || true
    sleep 2
    sudo kill -9 "${old_pid}" 2>/dev/null || true
  fi
fi

rm -f "${TBV21_DOCKER_SOCK}" "${TBV21_DOCKER_PID}"

echo "[INFO] starting tbv2.1 dockerd"
echo "[INFO] data-root=${TBV21_DOCKER_DATA_ROOT}"
echo "[INFO] persistent-cache-root=${TBV21_DOCKER_PERSISTENT_DATA_ROOT}"
echo "[INFO] exec-root=${TBV21_DOCKER_EXEC_ROOT}"
echo "[INFO] socket=${TBV21_DOCKER_SOCK}"

sudo env \
  HTTP_PROXY="${HTTP_PROXY}" HTTPS_PROXY="${HTTPS_PROXY}" NO_PROXY="${NO_PROXY}" \
  http_proxy="${http_proxy}" https_proxy="${https_proxy}" no_proxy="${no_proxy}" \
  nohup dockerd \
    --data-root "${TBV21_DOCKER_DATA_ROOT}" \
    --exec-root "${TBV21_DOCKER_EXEC_ROOT}" \
    --pidfile "${TBV21_DOCKER_PID}" \
    --host "unix://${TBV21_DOCKER_SOCK}" \
    --storage-driver vfs \
    --add-runtime "rootless-runc=${TBV21_HOME}/bin/runc-rootless.sh" \
    --default-runtime rootless-runc \
    --iptables=false \
    --ip-masq=false \
    --bridge=none \
    --default-address-pool base=10.244.0.0/16,size=24 \
    > "${TBV21_LOG_DIR}/docker/dockerd.log" 2>&1 &

for i in $(seq 1 60); do
  if [ -S "${TBV21_DOCKER_SOCK}" ]; then
    sudo chmod "${TBV21_DOCKER_SOCKET_MODE}" "${TBV21_DOCKER_SOCK}" 2>/dev/null || true
  fi
  if timeout 5 docker info >/dev/null 2>&1; then
    echo "[OK] tbv2.1 Docker ready after ${i}s"
    printf 'export DOCKER_HOST=%q\n' "${DOCKER_HOST}" > "${TBV21_DOCKER_HOST_ENV}"
    docker info | sed -n '1,100p'
    exit 0
  fi
  sleep 1
done

echo "[ERROR] tbv2.1 Docker did not become ready"
sed -n '1,240p' "${TBV21_LOG_DIR}/docker/dockerd.log" || true
exit 1
