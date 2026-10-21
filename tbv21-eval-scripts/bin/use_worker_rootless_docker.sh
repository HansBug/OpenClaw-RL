#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env.sh"

ACTION="${1:-start}"
SESSION="${TBV21_ROOTLESS_DOCKER_TMUX_SESSION:-tbv21_rootless_docker}"
ROOTLESS_LAUNCHER="${HOME}/.local/bin/dockerd-rootless-launch.sh"
ROOTLESS_ENV="${HOME}/.local/bin/docker-env.sh"
ROOTLESS_DATA_ROOT="${TBV21_ROOTLESS_DOCKER_DATA_ROOT:-${TBV21_HOME}/docker-data-rootless}"
ROOTLESS_LOG="${TBV21_LOG_DIR}/docker/rootless-worker-dockerd.log"
RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
SOCK="${RUNTIME_DIR}/docker.sock"

export XDG_RUNTIME_DIR="${RUNTIME_DIR}"
export DOCKER_HOST="unix://${SOCK}"
export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.45}"

need_rootless_install() {
  if [ ! -x "${ROOTLESS_LAUNCHER}" ] || [ ! -f "${ROOTLESS_ENV}" ]; then
    echo "[ERROR] rootless docker launcher is missing." >&2
    echo "Run first: bash ${SCRIPT_DIR}/setup_worker_rootless_docker.sh" >&2
    exit 1
  fi
}

write_rootless_config() {
  mkdir -p "${ROOTLESS_DATA_ROOT}" "${TBV21_LOG_DIR}/docker" "${TBV21_STATE_DIR}" "${HOME}/.config/docker"
  python3 - "${HOME}/.config/docker/daemon.json" "${ROOTLESS_DATA_ROOT}" <<'PY'
import json
import os
import sys

config = {
    "storage-driver": "fuse-overlayfs",
    "data-root": sys.argv[2],
}

mirrors = [
    item.strip()
    for item in os.environ.get("TBV21_DOCKER_REGISTRY_MIRRORS", "").split(",")
    if item.strip()
]
if mirrors:
    config["registry-mirrors"] = mirrors

proxies = {}
if os.environ.get("HTTP_PROXY"):
    proxies["http-proxy"] = os.environ["HTTP_PROXY"]
if os.environ.get("HTTPS_PROXY"):
    proxies["https-proxy"] = os.environ["HTTPS_PROXY"]
if os.environ.get("NO_PROXY"):
    proxies["no-proxy"] = os.environ["NO_PROXY"]
if proxies:
    config["proxies"] = proxies

with open(sys.argv[1], "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
PY
}

docker_ping() {
  [ -S "${SOCK}" ] && curl -fsS --max-time 2 --unix-socket "${SOCK}" http://localhost/_ping >/dev/null 2>&1
}

start_docker() {
  need_rootless_install
  write_rootless_config
  mkdir -p "${RUNTIME_DIR}"
  chmod 700 "${RUNTIME_DIR}" 2>/dev/null || true

  if docker_ping; then
    echo "[OK] rootless Docker already ready: ${DOCKER_HOST}"
    printf 'export DOCKER_HOST=%q\nexport DOCKER_API_VERSION=%q\n' "${DOCKER_HOST}" "${DOCKER_API_VERSION}" > "${TBV21_DOCKER_HOST_ENV}"
    docker info | sed -n '1,80p'
    return 0
  fi

  if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    rm -f "${SOCK}"
    tmux new-session -d -s "${SESSION}" \
      "export XDG_RUNTIME_DIR='${RUNTIME_DIR}' DOCKER_HOST='${DOCKER_HOST}' DOCKER_API_VERSION='${DOCKER_API_VERSION}' PATH='${HOME}/.local/bin:${PATH}'; exec '${ROOTLESS_LAUNCHER}' >> '${ROOTLESS_LOG}' 2>&1"
    echo "[INFO] tmux session=${SESSION}"
  else
    rm -f "${SOCK}"
    nohup env XDG_RUNTIME_DIR="${RUNTIME_DIR}" DOCKER_HOST="${DOCKER_HOST}" DOCKER_API_VERSION="${DOCKER_API_VERSION}" PATH="${HOME}/.local/bin:${PATH}" \
      "${ROOTLESS_LAUNCHER}" >> "${ROOTLESS_LOG}" 2>&1 &
  fi

  for i in $(seq 1 60); do
    if docker_ping; then
      echo "[OK] rootless Docker ready after ${i}s"
      printf 'export DOCKER_HOST=%q\nexport DOCKER_API_VERSION=%q\n' "${DOCKER_HOST}" "${DOCKER_API_VERSION}" > "${TBV21_DOCKER_HOST_ENV}"
      docker info | sed -n '1,100p'
      return 0
    fi
    if command -v tmux >/dev/null 2>&1 && ! tmux has-session -t "${SESSION}" 2>/dev/null; then
      echo "[ERROR] rootless Docker tmux session exited early" >&2
      tail -n 160 "${ROOTLESS_LOG}" || true
      exit 1
    fi
    sleep 1
  done

  echo "[ERROR] rootless Docker did not become ready" >&2
  tail -n 200 "${ROOTLESS_LOG}" || true
  exit 1
}

stop_docker() {
  if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
  fi
  if [ -f "${ROOTLESS_ENV}" ]; then
    # shellcheck disable=SC1090
    source "${ROOTLESS_ENV}"
    docker-down 2>/dev/null || true
  else
    pkill -f dockerd-rootless 2>/dev/null || true
    pkill -f rootlesskit 2>/dev/null || true
    pkill -x dockerd 2>/dev/null || true
  fi
  rm -f "${SOCK}"
  echo "[OK] rootless Docker stopped"
}

status_docker() {
  if docker_ping; then
    echo "[OK] rootless Docker reachable: ${DOCKER_HOST}"
    docker info | sed -n '1,100p'
  else
    echo "[WARN] rootless Docker is not reachable: ${DOCKER_HOST}"
    exit 1
  fi
}

smoke_docker() {
  start_docker
  echo "[INFO] docker run hello-world"
  docker run --rm hello-world | sed -n '1,40p'
  echo "[INFO] docker run alpine wget example.com"
  docker run --rm alpine wget --timeout=10 -qO- http://example.com | grep -o 'Example Domain' | head -1
  echo "[OK] rootless Docker smoke passed"
}

case "${ACTION}" in
  start) start_docker ;;
  stop) stop_docker ;;
  status) status_docker ;;
  smoke) smoke_docker ;;
  *)
    echo "Usage: $0 {start|stop|status|smoke}" >&2
    exit 2
    ;;
esac
