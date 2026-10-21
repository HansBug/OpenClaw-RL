#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env.sh"

ROOTLESSKIT="${TBV21_HOME}/bin/rootlesskit"
if [ ! -x "${ROOTLESSKIT}" ]; then
  echo "[ERROR] rootlesskit missing: ${ROOTLESSKIT}" >&2
  exit 1
fi
export PATH="${TBV21_HOME}/bin:${PATH}"

mkdir -p "${TBV21_DOCKER_DATA_ROOT}" "${TBV21_DOCKER_EXEC_ROOT}" "${TBV21_LOG_DIR}/docker" "${TBV21_STATE_DIR}" "${XDG_RUNTIME_DIR:-/tmp/tbv21-rootless-runtime-${USER:-user}}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/tbv21-rootless-runtime-${USER:-user}}"
chmod 700 "${XDG_RUNTIME_DIR}" 2>/dev/null || true

if [ -S "${TBV21_DOCKER_SOCK}" ] && timeout 5 docker info >/dev/null 2>&1; then
  echo "[OK] rootless tbv2.1 Docker already ready: ${DOCKER_HOST}"
  docker info | sed -n '1,100p'
  exit 0
fi

rm -f "${TBV21_DOCKER_SOCK}" "${TBV21_DOCKER_PID}"

echo "[INFO] starting rootless tbv2.1 dockerd"
echo "[INFO] data-root=${TBV21_DOCKER_DATA_ROOT}"
echo "[INFO] persistent-cache-root=${TBV21_DOCKER_PERSISTENT_DATA_ROOT}"
echo "[INFO] socket=${TBV21_DOCKER_SOCK}"

HTTP_PROXY="${HTTP_PROXY}" HTTPS_PROXY="${HTTPS_PROXY}" NO_PROXY="${NO_PROXY}" \
http_proxy="${http_proxy}" https_proxy="${https_proxy}" no_proxy="${no_proxy}" \
nohup "${ROOTLESSKIT}" \
  --net="${TBV21_ROOTLESSKIT_NET:-host}" \
  ${TBV21_ROOTLESSKIT_MTU:+--mtu="${TBV21_ROOTLESSKIT_MTU}"} \
  --copy-up=/etc \
  --disable-host-loopback \
  dockerd \
    --data-root "${TBV21_DOCKER_DATA_ROOT}" \
    --exec-root "${TBV21_DOCKER_EXEC_ROOT}" \
    --pidfile "${TBV21_DOCKER_PID}" \
    --host "unix://${TBV21_DOCKER_SOCK}" \
    --storage-driver fuse-overlayfs \
    --iptables=false \
    --ip-masq=false \
    --bridge=none \
    > "${TBV21_LOG_DIR}/docker/rootless-dockerd.log" 2>&1 &

for i in $(seq 1 90); do
  if [ -S "${TBV21_DOCKER_SOCK}" ]; then
    chmod "${TBV21_DOCKER_SOCKET_MODE}" "${TBV21_DOCKER_SOCK}" 2>/dev/null || true
  fi
  if timeout 5 docker info >/dev/null 2>&1; then
    echo "[OK] rootless tbv2.1 Docker ready after ${i}s"
    printf 'export DOCKER_HOST=%q\n' "${DOCKER_HOST}" > "${TBV21_DOCKER_HOST_ENV}"
    docker info | sed -n '1,120p'
    exit 0
  fi
  sleep 1
done

echo "[ERROR] rootless tbv2.1 Docker did not become ready"
sed -n '1,240p' "${TBV21_LOG_DIR}/docker/rootless-dockerd.log" || true
exit 1
