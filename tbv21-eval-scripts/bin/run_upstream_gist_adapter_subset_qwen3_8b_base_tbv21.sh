#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TBV21_MODEL_NAME="${TBV21_MODEL_NAME:-qwen3-8b-base}"
export TBV21_MODEL_PATH="${TBV21_MODEL_PATH:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
export TBV21_SGLANG_MEM_FRACTION="${TBV21_SGLANG_MEM_FRACTION:-0.6}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env.sh"

mkdir -p "${TBV21_LOG_DIR}" "${TBV21_JOBS_DIR}" "${TBV21_STATE_DIR}" "${TBV21_HOME}/runtime/apt"

if [ "${TBV21_UPSTREAM_GIST_ALL_TASKS:-0}" = "1" ]; then
  mapfile -t TASKS < <(find "${TBV21_TASKS_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort)
elif [ "$#" -gt 0 ]; then
  TASKS=("$@")
else
  TASKS=("hf-model-inference" "modernize-scientific-stack")
fi

TS="$(date +%Y%m%d_%H%M%S)"
TASK_TAG="$(IFS=_; echo "${TASKS[*]}")"
JOB_NAME="${TBV21_UPSTREAM_GIST_JOB_NAME:-upstream_gist_modeb_${TBV21_MODEL_NAME}_${TASK_TAG}_${TS}}"
LOG_FILE="${TBV21_LOG_DIR}/${JOB_NAME}.log"
STATE_FILE="${TBV21_STATE_DIR}/${JOB_NAME}.env"
CONFIG_FILE="${TBV21_STATE_DIR}/${JOB_NAME}.config.json"
N_ATTEMPTS="${TBV21_UPSTREAM_GIST_ATTEMPTS:-3}"
N_CONCURRENT="${TBV21_UPSTREAM_GIST_CONCURRENCY:-1}"
MAX_RETRIES="${TBV21_UPSTREAM_GIST_MAX_RETRIES:-0}"
TIMEOUT_MULTIPLIER="${TBV21_UPSTREAM_GIST_TIMEOUT_MULTIPLIER:-1.0}"
DELETE_ENV="${TBV21_UPSTREAM_GIST_DELETE_ENV:-1}"

TMUX_RUNTIME_DIR="${TBV21_TMUX_RUNTIME_DIR:-${TBV21_HOME}/runtime/tmux_runtime}"
APT_RUNTIME_DIR="${TBV21_APT_RUNTIME_DIR:-${TBV21_HOME}/runtime/apt}"
TOOLS_RUNTIME_DIR="${TBV21_TOOLS_RUNTIME_DIR:-${TBV21_HOME}/runtime/tools}"
WHEELHOUSE_DIR="${TBV21_WHEELHOUSE_DIR:-${TBV21_HOME}/runtime/wheelhouse}"
ADAPTER_DIR="${OPENCLAW_UPSTREAM_GIST_ADAPTER_DIR:-${TBV21_HOME}/runtime/openclaw_camel_adapter_upstream_gist}"
COMPAT_DIR="${OPENCLAW_UPSTREAM_GIST_COMPAT_DIR:-${TBV21_HOME}/runtime/openclaw_camel_adapter_upstream_compat}"
REAL_TERMINAL_RL_DIR="${REAL_OPENCLAW_TERMINAL_RL_DIR:-/mnt/shared-storage-user/narmodel/zhangshaoang/openclaw_rl_puyuan_3573bc3e_git/terminal-rl}"
VENDOR_PYTHONPATH_DIR="${OPENCLAW_UPSTREAM_GIST_VENDOR_PYTHONPATH_DIR:-${TBV21_HOME}/runtime/pythonpath_vendor_min}"
POOL1_SITE_PACKAGES="${OPENCLAW_POOL1_SITE_PACKAGES:-/mnt/shared-storage-user/tangjia/miniconda3/envs/openclaw-terminal-pool1/lib/python3.12/site-packages}"
HARBOR_PYTHON="${TBV21_HARBOR_PYTHON:-/mnt/shared-storage-user/tangjia/miniconda3/envs/terminal-bench/bin/python}"
CONTAINER_BASE_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
AGENT_CONTAINER_PATH="/opt/tbv21-tools/bin:/opt/tbv21-tmux/bin:${CONTAINER_BASE_PATH}"
ENV_CONTAINER_PATH="/opt/tbv21-tools/bin:${AGENT_CONTAINER_PATH}"
USE_COMPAT_OVERLAY="${OPENCLAW_UPSTREAM_GIST_USE_COMPAT_OVERLAY:-0}"
if [ "${USE_COMPAT_OVERLAY}" = "1" ]; then
  OPENCLAW_TERMINAL_RL_PATH="${COMPAT_DIR}"
  HOST_PYTHONPATH="${VENDOR_PYTHONPATH_DIR}:${ADAPTER_DIR}:${COMPAT_DIR}:${REAL_TERMINAL_RL_DIR}:${POOL1_SITE_PACKAGES}:${PYTHONPATH:-}"
else
  OPENCLAW_TERMINAL_RL_PATH="${REAL_TERMINAL_RL_DIR}"
  HOST_PYTHONPATH="${VENDOR_PYTHONPATH_DIR}:${ADAPTER_DIR}:${REAL_TERMINAL_RL_DIR}:${POOL1_SITE_PACKAGES}:${PYTHONPATH:-}"
fi

export OPENCLAW_TERMINAL_RL_DIR="${OPENCLAW_TERMINAL_RL_PATH}"
export REAL_OPENCLAW_TERMINAL_RL_DIR="${REAL_TERMINAL_RL_DIR}"
export PYTHONPATH="${HOST_PYTHONPATH}"

{
  printf 'export TBV21_UPSTREAM_GIST_JOB_NAME=%q\n' "${JOB_NAME}"
  printf 'export TBV21_UPSTREAM_GIST_LOG_FILE=%q\n' "${LOG_FILE}"
  printf 'export TBV21_UPSTREAM_GIST_CONFIG_FILE=%q\n' "${CONFIG_FILE}"
  printf 'export TBV21_UPSTREAM_GIST_TASKS=%q\n' "${TASKS[*]}"
  printf 'export OPENCLAW_TERMINAL_RL_DIR=%q\n' "${OPENCLAW_TERMINAL_RL_DIR}"
  printf 'export REAL_OPENCLAW_TERMINAL_RL_DIR=%q\n' "${REAL_OPENCLAW_TERMINAL_RL_DIR}"
  printf 'export PYTHONPATH=%q\n' "${PYTHONPATH}"
} > "${STATE_FILE}"

touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

log() {
  echo "[$(date '+%F %T')] $*"
}

log "job=${JOB_NAME}"
log "log=${LOG_FILE}"
log "config=${CONFIG_FILE}"
log "model=${TBV21_MODEL_NAME}"
log "model_path=${TBV21_MODEL_PATH}"
log "tasks=${TASKS[*]}"
log "attempts=${N_ATTEMPTS} concurrency=${N_CONCURRENT} retries=${MAX_RETRIES} delete_env=${DELETE_ENV}"
log "adapter_dir=${ADAPTER_DIR}"
log "real_terminal_rl_dir=${REAL_TERMINAL_RL_DIR}"
log "vendor_pythonpath_dir=${VENDOR_PYTHONPATH_DIR}"
log "openclaw_terminal_rl_dir=${OPENCLAW_TERMINAL_RL_DIR}"
log "use_compat_overlay=${USE_COMPAT_OVERLAY}"

if [ ! -d "${TBV21_TASKS_DIR}" ]; then
  log "[ERROR] tasks dir missing: ${TBV21_TASKS_DIR}"
  exit 1
fi
if [ ! -d "${TBV21_MODEL_PATH}" ]; then
  log "[ERROR] model path missing: ${TBV21_MODEL_PATH}"
  exit 1
fi
if [ ! -f "${ADAPTER_DIR}/openclaw_camel_adapter.py" ]; then
  log "[ERROR] adapter missing: ${ADAPTER_DIR}/openclaw_camel_adapter.py"
  exit 1
fi
if [ ! -d "${VENDOR_PYTHONPATH_DIR}" ]; then
  log "[ERROR] vendor pythonpath dir missing: ${VENDOR_PYTHONPATH_DIR}"
  exit 1
fi
if [ "${USE_COMPAT_OVERLAY}" = "1" ] && { [ ! -f "${COMPAT_DIR}/inference_client.py" ] || [ ! -f "${COMPAT_DIR}/agent_runner.py" ]; }; then
  log "[ERROR] compat overlay incomplete: ${COMPAT_DIR}"
  exit 1
fi
if [ ! -d "${REAL_TERMINAL_RL_DIR}" ]; then
  log "[ERROR] real OpenClaw terminal-rl dir missing: ${REAL_TERMINAL_RL_DIR}"
  exit 1
fi
if [ ! -f "${OPENCLAW_TERMINAL_RL_DIR}/inference_client.py" ] || [ ! -f "${OPENCLAW_TERMINAL_RL_DIR}/agent_runner.py" ]; then
  log "[ERROR] OPENCLAW_TERMINAL_RL_DIR lacks expected terminal-rl API files: ${OPENCLAW_TERMINAL_RL_DIR}"
  exit 1
fi
for task in "${TASKS[@]}"; do
  if [ ! -d "${TBV21_TASKS_DIR}/${task}" ]; then
    log "[ERROR] task missing: ${TBV21_TASKS_DIR}/${task}"
    exit 1
  fi
done

sha256sum "${ADAPTER_DIR}/openclaw_camel_adapter.py" "${OPENCLAW_TERMINAL_RL_DIR}/inference_client.py" "${OPENCLAW_TERMINAL_RL_DIR}/agent_runner.py" || true

cat > "${APT_RUNTIME_DIR}/apt.conf" <<EOF
Acquire::http::Proxy "${HTTP_PROXY}";
Acquire::https::Proxy "${HTTPS_PROXY}";
Acquire::Retries "3";
Acquire::http::Timeout "30";
Acquire::https::Timeout "30";
EOF

log "starting rootless Docker backend"
"${SCRIPT_DIR}/use_worker_rootless_docker.sh" start
# shellcheck disable=SC1090
source "${TBV21_DOCKER_HOST_ENV}"
export DOCKER_HOST="${DOCKER_HOST:-${TBV21_ROOTLESS_DOCKER_HOST}}"
export DOCKER_CONFIG="${DOCKER_CONFIG:-${TBV21_DOCKER_CONFIG_DIR}}"
export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.45}"
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-0}"
export COMPOSE_DOCKER_CLI_BUILD="${COMPOSE_DOCKER_CLI_BUILD:-0}"
docker info --format 'server={{.ServerVersion}} root={{.DockerRootDir}} driver={{.Driver}} cgroup={{.CgroupDriver}} containers={{.Containers}} images={{.Images}}'

if [ "${TBV21_PREPULL_IMAGES:-1}" = "1" ]; then
  log "pre-pulling selected TBv2.1 task images"
  for task in "${TASKS[@]}"; do
    "${SCRIPT_DIR}/prepull_tbv21_images.sh" "${TBV21_TASKS_DIR}/${task}"
  done
fi

if ! curl --noproxy '*' -fsS --max-time 5 "http://${TBV21_SGLANG_HOST}:${TBV21_SGLANG_PORT}/v1/models" >/dev/null; then
  log "starting qwen3-8b-base modeB SGLang endpoint"
  "${SCRIPT_DIR}/start_sglang_qwen3_8b_base_modeb.sh" "${TBV21_GPU_IDS:-0,1,2,3}"
fi
log "checking SGLang endpoint"
curl --noproxy '*' -fsS --max-time 10 "http://${TBV21_SGLANG_HOST}:${TBV21_SGLANG_PORT}/v1/models"
echo

TASKS_JSON="$(python3 - "${TASKS[@]}" <<'PY'
import json, sys
print(json.dumps(sys.argv[1:]))
PY
)"

PYTHONPATH= python3 - "${CONFIG_FILE}" "${JOB_NAME}" "${TASKS_JSON}" "${TMUX_RUNTIME_DIR}" "${APT_RUNTIME_DIR}" "${TOOLS_RUNTIME_DIR}" "${WHEELHOUSE_DIR}" "${ADAPTER_DIR}" "${ENV_CONTAINER_PATH}" "${AGENT_CONTAINER_PATH}" "${N_ATTEMPTS}" "${N_CONCURRENT}" "${MAX_RETRIES}" "${TIMEOUT_MULTIPLIER}" "${DELETE_ENV}" "${HOST_PYTHONPATH}" "${OPENCLAW_TERMINAL_RL_DIR}" "${REAL_TERMINAL_RL_DIR}" <<'PY'
import json
import os
import sys

(
    config_file,
    job_name,
    tasks_json,
    tmux_runtime,
    apt_runtime,
    tools_runtime,
    wheelhouse,
    adapter_dir,
    env_container_path,
    agent_container_path,
    n_attempts,
    n_concurrent,
    max_retries,
    timeout_multiplier,
    delete_env,
    host_pythonpath,
    openclaw_terminal_rl_dir,
    real_terminal_rl_dir,
) = sys.argv[1:19]

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

docker_env = {}
for key in (
    "DOCKER_HOST",
    "DOCKER_API_VERSION",
    "DOCKER_CONFIG",
    "DOCKER_BUILDKIT",
    "COMPOSE_DOCKER_CLI_BUILD",
):
    if os.environ.get(key):
        docker_env[key] = os.environ[key]

agent_env = {
    "OPENAI_API_KEY": "dummy",
    "OPENAI_BASE_URL": f"http://{os.environ['TBV21_SGLANG_HOST']}:{os.environ['TBV21_SGLANG_PORT']}/v1",
    "SGLANG_BASE_URL": f"http://{os.environ['TBV21_SGLANG_HOST']}:{os.environ['TBV21_SGLANG_PORT']}",
    "OPENCLAW_TERMINAL_RL_DIR": openclaw_terminal_rl_dir,
    "REAL_OPENCLAW_TERMINAL_RL_DIR": real_terminal_rl_dir,
    "PYTHONPATH": host_pythonpath,
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
    **docker_env,
}

config = {
    "job_name": job_name,
    "jobs_dir": os.environ["TBV21_JOBS_DIR"],
    "n_attempts": int(n_attempts),
    "timeout_multiplier": float(timeout_multiplier),
    "n_concurrent_trials": int(n_concurrent),
    "quiet": True,
    "retry": {"max_retries": int(max_retries)},
    "environment": {
        "type": "docker",
        "delete": bool(int(delete_env)),
        "mounts_json": [
            {"type": "bind", "source": tmux_runtime, "target": "/opt/tbv21-tmux", "read_only": True, "bind": {"create_host_path": False}},
            {"type": "bind", "source": apt_runtime, "target": "/opt/tbv21-apt", "read_only": True, "bind": {"create_host_path": False}},
            {"type": "bind", "source": tools_runtime, "target": "/opt/tbv21-tools", "read_only": True, "bind": {"create_host_path": False}},
            {"type": "bind", "source": wheelhouse, "target": "/opt/tbv21-wheelhouse", "read_only": True, "bind": {"create_host_path": False}},
        ],
        "env": env,
    },
    "agents": [
        {
            "import_path": "openclaw_camel_adapter:OpenClawCamelAgent",
            "model_name": f"openai/{os.environ['TBV21_MODEL_NAME']}",
            "kwargs": {
                "sglang_url": f"http://{os.environ['TBV21_SGLANG_HOST']}:{os.environ['TBV21_SGLANG_PORT']}",
                "sglang_served_name": os.environ["TBV21_MODEL_NAME"],
                "hf_model_dir": os.environ["TBV21_MODEL_PATH"],
                "model_type": "Qwen3",
                "max_iteration": 10,
                "max_total_tokens": 16384,
                "max_parse_errors": 3,
                "non_think_mode": False,
                "tool_call_parser": "qwen25",
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": 8192,
                "rollout_seed": 42,
                "rollout_skip_special_tokens": False,
                "terminal_toolkit_timeout": 20.0,
                "request_timeout_sec": 600.0,
                "sglang_max_retries": 30,
            },
            "env": agent_env,
        }
    ],
    "datasets": [
        {
            "path": os.environ["TBV21_TASKS_DIR"],
            "task_names": json.loads(tasks_json),
        }
    ],
}
with open(config_file, "w") as f:
    json.dump(config, f, indent=2)
PY

log "validating upstream gist adapter import and generated config"
"${HARBOR_PYTHON}" - <<'PY'
import importlib
import os
mod = importlib.import_module("openclaw_camel_adapter")
cls = mod.OpenClawCamelAgent
print("[OK] adapter", cls.name())
print("[OK] adapter_file", mod.__file__)
print("[OK] OPENCLAW_TERMINAL_RL_DIR", os.environ.get("OPENCLAW_TERMINAL_RL_DIR"))
PY
python3 -m json.tool "${CONFIG_FILE}" >/dev/null

log "starting Harbor upstream gist adapter eval"
env \
  PYTHONPATH="${PYTHONPATH}" \
  OPENCLAW_TERMINAL_RL_DIR="${OPENCLAW_TERMINAL_RL_DIR}" \
  REAL_OPENCLAW_TERMINAL_RL_DIR="${REAL_OPENCLAW_TERMINAL_RL_DIR}" \
  DOCKER_HOST="${DOCKER_HOST}" \
  DOCKER_API_VERSION="${DOCKER_API_VERSION}" \
  DOCKER_CONFIG="${DOCKER_CONFIG}" \
  DOCKER_BUILDKIT="${DOCKER_BUILDKIT}" \
  COMPOSE_DOCKER_CLI_BUILD="${COMPOSE_DOCKER_CLI_BUILD}" \
  "${TBV21_HARBOR_BIN}" run --config "${CONFIG_FILE}"

log "Harbor upstream gist adapter eval completed"
"${SCRIPT_DIR}/summarize_harbor_job.py" "${TBV21_JOBS_DIR}/${JOB_NAME}" || true
