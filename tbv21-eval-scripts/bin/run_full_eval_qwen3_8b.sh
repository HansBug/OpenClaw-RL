#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env.sh"

mkdir -p "${TBV21_LOG_DIR}" "${TBV21_JOBS_DIR}" "${TBV21_STATE_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
JOB_NAME="${TBV21_FULL_EVAL_JOB_NAME:-full_eval_tbv21_${TBV21_MODEL_NAME}_${TS}}"
LOG_FILE="${TBV21_LOG_DIR}/${JOB_NAME}.log"
STATE_FILE="${TBV21_STATE_DIR}/${JOB_NAME}.env"
GPU_IDS="${TBV21_GPU_IDS:-0,1,2,3}"
N_CONCURRENT="${TBV21_FULL_EVAL_CONCURRENCY:-2}"
MAX_RETRIES="${TBV21_FULL_EVAL_MAX_RETRIES:-1}"
TIMEOUT_MULTIPLIER="${TBV21_FULL_EVAL_TIMEOUT_MULTIPLIER:-1.0}"
TMUX_RUNTIME_DIR="${TBV21_TMUX_RUNTIME_DIR:-${TBV21_HOME}/runtime/tmux_runtime}"
APT_RUNTIME_DIR="${TBV21_APT_RUNTIME_DIR:-${TBV21_HOME}/runtime/apt}"
TOOLS_RUNTIME_DIR="${TBV21_TOOLS_RUNTIME_DIR:-${TBV21_HOME}/runtime/tools}"
WHEELHOUSE_DIR="${TBV21_WHEELHOUSE_DIR:-${TBV21_HOME}/runtime/wheelhouse}"
CONTAINER_BASE_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
AGENT_CONTAINER_PATH="/opt/tbv21-tools/bin:/opt/tbv21-tmux/bin:${CONTAINER_BASE_PATH}"
ENV_CONTAINER_PATH="/opt/tbv21-tools/bin:${AGENT_CONTAINER_PATH}"
CONFIG_FILE="${TBV21_STATE_DIR}/${JOB_NAME}.config.json"

{
  printf 'export TBV21_FULL_EVAL_JOB_NAME=%q\n' "${JOB_NAME}"
  printf 'export TBV21_FULL_EVAL_LOG_FILE=%q\n' "${LOG_FILE}"
  printf 'export TBV21_FULL_EVAL_CONCURRENCY=%q\n' "${N_CONCURRENT}"
  printf 'export TBV21_FULL_EVAL_MAX_RETRIES=%q\n' "${MAX_RETRIES}"
  printf 'export TBV21_FULL_EVAL_TIMEOUT_MULTIPLIER=%q\n' "${TIMEOUT_MULTIPLIER}"
} > "${STATE_FILE}"

touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

log() {
  echo "[$(date '+%F %T')] $*"
}

log "job=${JOB_NAME}"
log "log=${LOG_FILE}"
log "state=${STATE_FILE}"
log "model=${TBV21_MODEL_NAME}"
log "model_path=${TBV21_MODEL_PATH}"
log "tasks_dir=${TBV21_TASKS_DIR}"
log "n_concurrent=${N_CONCURRENT} max_retries=${MAX_RETRIES} timeout_multiplier=${TIMEOUT_MULTIPLIER}"
log "tmux_runtime=${TMUX_RUNTIME_DIR}"
log "apt_runtime=${APT_RUNTIME_DIR}"
log "tools_runtime=${TOOLS_RUNTIME_DIR}"
log "wheelhouse=${WHEELHOUSE_DIR}"

if [ ! -d "${TBV21_TASKS_DIR}" ]; then
  log "[ERROR] tasks dir missing: ${TBV21_TASKS_DIR}"
  exit 1
fi
if [ ! -d "${TBV21_MODEL_PATH}" ]; then
  log "[ERROR] model path missing: ${TBV21_MODEL_PATH}"
  exit 1
fi

TASK_COUNT="$(find "${TBV21_TASKS_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l | awk '{print $1}')"
log "task_count=${TASK_COUNT}"

if [ ! -x "${TMUX_RUNTIME_DIR}/bin/tmux" ]; then
  log "[ERROR] tmux runtime missing: ${TMUX_RUNTIME_DIR}/bin/tmux"
  exit 1
fi
if [ ! -x "${TOOLS_RUNTIME_DIR}/bin/uv" ]; then
  log "[ERROR] tools runtime missing: ${TOOLS_RUNTIME_DIR}/bin/uv"
  exit 1
fi
if [ ! -d "${WHEELHOUSE_DIR}" ]; then
  log "[ERROR] wheelhouse missing: ${WHEELHOUSE_DIR}"
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
unset PYTHONPATH

log "starting rootless Docker backend"
if [ ! -x "${HOME}/.local/bin/dockerd-rootless-launch.sh" ]; then
  log "rootless Docker launcher missing; running setup"
  "${SCRIPT_DIR}/setup_worker_rootless_docker.sh"
fi
"${SCRIPT_DIR}/use_worker_rootless_docker.sh" start
# shellcheck disable=SC1090
source "${TBV21_DOCKER_HOST_ENV}"

log "Docker backend summary"
docker info --format 'server={{.ServerVersion}} root={{.DockerRootDir}} driver={{.Driver}} cgroup={{.CgroupDriver}} containers={{.Containers}} images={{.Images}}'
docker compose version

if [ "${TBV21_PREPULL_IMAGES:-1}" = "1" ]; then
  log "pre-pulling TBv2.1 task images"
  "${SCRIPT_DIR}/prepull_tbv21_images.sh" "${TBV21_TASKS_DIR}"
else
  log "skipping TBv2.1 image pre-pull because TBV21_PREPULL_IMAGES=${TBV21_PREPULL_IMAGES:-}"
fi

log "starting SGLang"
"${SCRIPT_DIR}/start_sglang.sh" "${GPU_IDS}"

log "SGLang models endpoint"
curl --noproxy '*' -fsS --max-time 10 "http://${TBV21_SGLANG_HOST}:${TBV21_SGLANG_PORT}/v1/models"
echo

log "GPU snapshot before Harbor"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

python - "${CONFIG_FILE}" "${JOB_NAME}" "${TMUX_RUNTIME_DIR}" "${APT_RUNTIME_DIR}" "${TOOLS_RUNTIME_DIR}" "${WHEELHOUSE_DIR}" "${ENV_CONTAINER_PATH}" "${AGENT_CONTAINER_PATH}" "${N_CONCURRENT}" "${MAX_RETRIES}" "${TIMEOUT_MULTIPLIER}" <<'PY'
import json
import os
import sys

(
    config_file,
    job_name,
    tmux_runtime,
    apt_runtime,
    tools_runtime,
    wheelhouse,
    env_container_path,
    agent_container_path,
    n_concurrent,
    max_retries,
    timeout_multiplier,
) = sys.argv[1:12]
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
    "timeout_multiplier": float(timeout_multiplier),
    "n_concurrent_trials": int(n_concurrent),
    "retry": {"max_retries": int(max_retries)},
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
        }
    ],
}
with open(config_file, "w") as f:
    json.dump(config, f, indent=2)
PY

log "validating generated Harbor config"
python - "${CONFIG_FILE}" <<'PY'
import json
import sys

config_file = sys.argv[1]
with open(config_file) as f:
    config = json.load(f)

env = config.get("environment", {}).get("env", {})
agent_env = (config.get("agents") or [{}])[0].get("env", {})
model_info = (config.get("agents") or [{}])[0].get("kwargs", {}).get("model_info", {})

required_env = {
    "DEBIAN_FRONTEND": "noninteractive",
    "TZ": "Etc/UTC",
    "APT_CONFIG": "/opt/tbv21-apt/apt.conf",
}
missing = []
for key, expected in required_env.items():
    if env.get(key) != expected:
        missing.append(f"environment.env.{key}={env.get(key)!r}")
    if agent_env.get(key) != expected:
        missing.append(f"agents[0].env.{key}={agent_env.get(key)!r}")
for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
    if key not in env:
        missing.append(f"environment.env.{key}=<missing>")
for key in ("max_input_tokens", "max_output_tokens"):
    if int(model_info.get(key) or 0) <= 0:
        missing.append(f"agents[0].kwargs.model_info.{key}={model_info.get(key)!r}")
if missing:
    raise SystemExit("invalid Harbor config: " + "; ".join(missing))
print("[OK] Harbor config has proxy env, noninteractive apt env, and model_info")
PY

log "Harbor config=${CONFIG_FILE}"
log "starting Harbor full eval"
set -x
"${TBV21_HARBOR_BIN}" run --config "${CONFIG_FILE}"
set +x

log "Harbor full eval completed"
log "result files"
find "${TBV21_JOBS_DIR}/${JOB_NAME}" -maxdepth 4 -type f | sort | sed -n '1,200p' || true
