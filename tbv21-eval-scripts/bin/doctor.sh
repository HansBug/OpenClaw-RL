#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env.sh"

echo "== paths =="
echo "TBV21_HOME=${TBV21_HOME}"
echo "TBV21_TASKS_DIR=${TBV21_TASKS_DIR}"
echo "TBV21_MODEL_PATH=${TBV21_MODEL_PATH}"
echo "TBV21_MODEL_REAL=$(readlink -f "${TBV21_MODEL_PATH}" 2>/dev/null || true)"
echo "TBV21_HARBOR_VENV=${TBV21_HARBOR_VENV}"
echo "TBV21_SGLANG_VENV=${TBV21_SGLANG_VENV}"
echo "TBV21_HARBOR_BIN=${TBV21_HARBOR_BIN}"
echo "TBV21_SGLANG_PYTHON=${TBV21_SGLANG_PYTHON}"
echo "DOCKER_HOST=${DOCKER_HOST}"
echo "DOCKER_CONFIG=${DOCKER_CONFIG}"

echo "== disk =="
df -h "${TBV21_HOME}" "${TBV21_DOCKER_DATA_ROOT}" "${TBV21_ROOTLESS_DOCKER_DATA_ROOT}" /tmp 2>/dev/null || true

echo "== gpu =="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true

echo "== docker =="
timeout 8 docker info 2>&1 | sed -n '1,120p' || true

echo "== dataset =="
if [ -d "${TBV21_TASKS_DIR}" ]; then
  echo "task dirs: $(find "${TBV21_TASKS_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
else
  echo "missing: ${TBV21_TASKS_DIR}"
fi

echo "== full eval static resources =="
check_path() {
  local path="$1"
  if [ -e "${path}" ]; then
    if [ -L "${path}" ]; then
      echo "ok symlink: ${path} -> $(readlink -f "${path}")"
    else
      echo "ok: ${path}"
    fi
  else
    echo "missing: ${path}"
  fi
}
check_exec() {
  local path="$1"
  if [ -x "${path}" ]; then
    echo "ok exec: ${path}"
  else
    echo "missing exec: ${path}"
  fi
}
check_exec "${TBV21_HOME}/runtime/tmux_runtime/bin/tmux"
check_exec "${TBV21_HOME}/runtime/tools/bin/uv"
check_exec "${TBV21_HOME}/docker-cli-plugins/docker-compose"
check_path "${TBV21_HOME}/runtime/wheelhouse"
check_path "${TBV21_HOME}/docker-cli/config.json"
if [ -r "$(readlink -f "${TBV21_MODEL_PATH}" 2>/dev/null)/config.json" ]; then
  echo "model config: readable"
else
  echo "model config: missing or unreadable"
fi
if [ -f "${TBV21_DOCKER_HOST_ENV}" ]; then
  echo "docker host env: ${TBV21_DOCKER_HOST_ENV}"
  cat "${TBV21_DOCKER_HOST_ENV}"
else
  echo "docker host env missing: ${TBV21_DOCKER_HOST_ENV}"
fi
if [ -f "${TBV21_HOME}/docker-cli/config.json" ]; then
  echo "docker cli proxy config:"
  sed -n '1,80p' "${TBV21_HOME}/docker-cli/config.json"
fi

echo "== python tools =="
if [ -x "${TBV21_HARBOR_BIN}" ]; then
  echo "-- harbor venv --"
  "${TBV21_HARBOR_BIN}" --version || true
  "${TBV21_HARBOR_BIN}" --help >/dev/null 2>&1 && echo "harbor: ok" || echo "harbor: missing or failing"
else
  echo "missing harbor executable: ${TBV21_HARBOR_BIN}"
fi
if [ -x "${TBV21_SGLANG_PYTHON}" ]; then
  echo "-- sglang venv --"
  "${TBV21_SGLANG_PYTHON}" --version
  "${TBV21_SGLANG_PYTHON}" - <<'PY' || true
import importlib.util
for name in ["sglang", "openai", "transformers"]:
    print(f"{name}: {bool(importlib.util.find_spec(name))}")
PY
else
  echo "missing sglang python: ${TBV21_SGLANG_PYTHON}"
fi
