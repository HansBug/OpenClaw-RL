#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env.sh"

mkdir -p "${TBV21_LOG_DIR}" "${TBV21_JOBS_DIR}" "${TBV21_STATE_DIR}"

TASK="${1:-gcode-to-text}"
TS="$(date +%Y%m%d_%H%M%S)"
JOB_NAME="${2:-one_task_${TASK}_${TBV21_MODEL_NAME}_${TS}}"
LOG_FILE="${TBV21_LOG_DIR}/${JOB_NAME}.log"
TMUX_RUNTIME_DIR="${TBV21_TMUX_RUNTIME_DIR:-${TBV21_HOME}/runtime/tmux_runtime}"
APT_RUNTIME_DIR="${TBV21_APT_RUNTIME_DIR:-${TBV21_HOME}/runtime/apt}"
TOOLS_RUNTIME_DIR="${TBV21_TOOLS_RUNTIME_DIR:-${TBV21_HOME}/runtime/tools}"
WHEELHOUSE_DIR="${TBV21_WHEELHOUSE_DIR:-${TBV21_HOME}/runtime/wheelhouse}"
CONTAINER_BASE_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
AGENT_CONTAINER_PATH="/opt/tbv21-tools/bin:/opt/tbv21-tmux/bin:${CONTAINER_BASE_PATH}"
ENV_CONTAINER_PATH="/opt/tbv21-tools/bin:${AGENT_CONTAINER_PATH}"
CONFIG_FILE="${TBV21_STATE_DIR}/${JOB_NAME}.config.json"

touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date '+%F %T')] job=${JOB_NAME}"
echo "[$(date '+%F %T')] task=${TASK}"
echo "[$(date '+%F %T')] log=${LOG_FILE}"
echo "[$(date '+%F %T')] tmux_runtime=${TMUX_RUNTIME_DIR}"
echo "[$(date '+%F %T')] apt_runtime=${APT_RUNTIME_DIR}"
echo "[$(date '+%F %T')] tools_runtime=${TOOLS_RUNTIME_DIR}"
echo "[$(date '+%F %T')] wheelhouse=${WHEELHOUSE_DIR}"

if [ ! -x "${TMUX_RUNTIME_DIR}/bin/tmux" ]; then
  echo "[ERROR] tmux runtime missing: ${TMUX_RUNTIME_DIR}/bin/tmux" >&2
  exit 1
fi
if [ ! -x "${TOOLS_RUNTIME_DIR}/bin/uv" ]; then
  echo "[ERROR] tools runtime missing: ${TOOLS_RUNTIME_DIR}/bin/uv" >&2
  exit 1
fi
if [ ! -d "${WHEELHOUSE_DIR}" ]; then
  echo "[ERROR] wheelhouse missing: ${WHEELHOUSE_DIR}" >&2
  exit 1
fi
mkdir -p "${APT_RUNTIME_DIR}"
cat > "${APT_RUNTIME_DIR}/apt.conf" <<EOF
Acquire::http::Proxy "${HTTP_PROXY}";
Acquire::https::Proxy "${HTTPS_PROXY}";
Acquire::Retries "3";
Acquire::http::Timeout "30";
Acquire::https::Timeout "30";
EOF

if [ ! -x "${HOME}/.local/bin/dockerd-rootless-launch.sh" ]; then
  echo "[$(date '+%F %T')] rootless Docker launcher missing; running setup"
  "${SCRIPT_DIR}/setup_worker_rootless_docker.sh"
fi
"${SCRIPT_DIR}/use_worker_rootless_docker.sh" start
# shellcheck disable=SC1090
source "${TBV21_DOCKER_HOST_ENV}"

if ! curl --noproxy '*' -fsS --max-time 5 "http://${TBV21_SGLANG_HOST}:${TBV21_SGLANG_PORT}/v1/models" >/dev/null; then
  "${SCRIPT_DIR}/start_sglang.sh" "${TBV21_GPU_IDS:-0,1,2,3}"
fi

unset PYTHONPATH

python - "${CONFIG_FILE}" "${JOB_NAME}" "${TASK}" "${TMUX_RUNTIME_DIR}" "${APT_RUNTIME_DIR}" "${TOOLS_RUNTIME_DIR}" "${WHEELHOUSE_DIR}" "${ENV_CONTAINER_PATH}" "${AGENT_CONTAINER_PATH}" <<'PY'
import json
import os
import sys

(
    config_file,
    job_name,
    task,
    tmux_runtime,
    apt_runtime,
    tools_runtime,
    wheelhouse,
    env_container_path,
    agent_container_path,
) = sys.argv[1:10]
env = {
    "PATH": env_container_path,
    "APT_CONFIG": "/opt/tbv21-apt/apt.conf",
    "DEBIAN_FRONTEND": "noninteractive",
    "TZ": "Etc/UTC",
    "UV_LINK_MODE": "copy",
    "UV_PYTHON_INSTALL_DIR": "/tmp/tbv21-uv-python",
    "UV_CACHE_DIR": "/tmp/tbv21-uv-cache",
    "UV_FIND_LINKS": "/opt/tbv21-wheelhouse",
    "UV_INDEX_URL": "http://mirrors.h.pjlab.org.cn/pypi/simple/",
    "UV_EXTRA_INDEX_URL": "http://pypi.i.h.pjlab.org.cn/brain/dev/+simple",
    "UV_INDEX_STRATEGY": "unsafe-best-match",
    "UV_TRUSTED_HOST": "mirrors.h.pjlab.org.cn pypi.i.h.pjlab.org.cn",
}
for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"):
    if os.environ.get(key):
        env[key] = os.environ[key]

config = {
    "job_name": job_name,
    "jobs_dir": os.environ["TBV21_JOBS_DIR"],
    "n_attempts": 1,
    "timeout_multiplier": float(os.environ.get("TBV21_ONE_TASK_TIMEOUT_MULTIPLIER", "1.0")),
    "n_concurrent_trials": 1,
    "retry": {"max_retries": 0},
    "environment": {
        "type": "docker",
        "delete": True,
        "mounts_json": [
            {
                "type": "bind",
                "source": tmux_runtime,
                "target": "/opt/tbv21-tmux",
                "read_only": True,
                "bind": {"create_host_path": False},
            },
            {
                "type": "bind",
                "source": apt_runtime,
                "target": "/opt/tbv21-apt",
                "read_only": True,
                "bind": {"create_host_path": False},
            },
            {
                "type": "bind",
                "source": tools_runtime,
                "target": "/opt/tbv21-tools",
                "read_only": True,
                "bind": {"create_host_path": False},
            },
            {
                "type": "bind",
                "source": wheelhouse,
                "target": "/opt/tbv21-wheelhouse",
                "read_only": True,
                "bind": {"create_host_path": False},
            }
        ],
        "env": env,
    },
    "agents": [
        {
            "name": "terminus-2",
            "model_name": f"openai/{os.environ['TBV21_MODEL_NAME']}",
            "kwargs": {
                "api_base": f"http://{os.environ['TBV21_SGLANG_HOST']}:{os.environ['TBV21_SGLANG_PORT']}/v1",
                "record_terminal_session": False,
                "model_info": {
                    "max_input_tokens": int(os.environ["TBV21_MODEL_MAX_INPUT_TOKENS"]),
                    "max_output_tokens": int(os.environ["TBV21_MODEL_MAX_OUTPUT_TOKENS"]),
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                },
            },
            "env": {
                "OPENAI_API_KEY": "dummy",
                "OPENAI_BASE_URL": f"http://{os.environ['TBV21_SGLANG_HOST']}:{os.environ['TBV21_SGLANG_PORT']}/v1",
            "PATH": agent_container_path,
            "APT_CONFIG": "/opt/tbv21-apt/apt.conf",
            "DEBIAN_FRONTEND": "noninteractive",
            "TZ": "Etc/UTC",
            "UV_LINK_MODE": "copy",
            "UV_PYTHON_INSTALL_DIR": "/tmp/tbv21-uv-python",
            "UV_CACHE_DIR": "/tmp/tbv21-uv-cache",
            "UV_FIND_LINKS": "/opt/tbv21-wheelhouse",
            "UV_INDEX_URL": "http://mirrors.h.pjlab.org.cn/pypi/simple/",
            "UV_EXTRA_INDEX_URL": "http://pypi.i.h.pjlab.org.cn/brain/dev/+simple",
            "UV_INDEX_STRATEGY": "unsafe-best-match",
            "UV_TRUSTED_HOST": "mirrors.h.pjlab.org.cn pypi.i.h.pjlab.org.cn",
        },
        }
    ],
    "datasets": [
        {
            "path": os.environ["TBV21_TASKS_DIR"],
            "task_names": [task],
        }
    ],
}
with open(config_file, "w") as f:
    json.dump(config, f, indent=2)
PY

echo "[$(date '+%F %T')] config=${CONFIG_FILE}"

set -x
"${TBV21_HARBOR_BIN}" run --config "${CONFIG_FILE}"
set +x

echo "[$(date '+%F %T')] one task eval completed"
find "${TBV21_JOBS_DIR}/${JOB_NAME}" -maxdepth 4 -type f | sort | sed -n '1,200p' || true
