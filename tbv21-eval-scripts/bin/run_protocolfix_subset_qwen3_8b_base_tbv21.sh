#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TBV21_MODEL_NAME="${TBV21_MODEL_NAME:-qwen3-8b-base}"
export TBV21_MODEL_PATH="${TBV21_MODEL_PATH:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
export TBV21_SGLANG_MEM_FRACTION="${TBV21_SGLANG_MEM_FRACTION:-0.6}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env.sh"

mkdir -p "${TBV21_LOG_DIR}" "${TBV21_JOBS_DIR}" "${TBV21_STATE_DIR}" "${TBV21_HOME}/runtime/apt"

if [ "$#" -gt 0 ]; then
  TASKS=("$@")
else
  TASKS=("hf-model-inference" "modernize-scientific-stack")
fi

TS="$(date +%Y%m%d_%H%M%S)"
TASK_TAG="$(IFS=_; echo "${TASKS[*]}")"
JOB_NAME="${TBV21_PROTOCOLFIX_JOB_NAME:-protocolfix1_${TBV21_MODEL_NAME}_${TASK_TAG}_${TS}}"
LOG_FILE="${TBV21_LOG_DIR}/${JOB_NAME}.log"
STATE_FILE="${TBV21_STATE_DIR}/${JOB_NAME}.env"
CONFIG_FILE="${TBV21_STATE_DIR}/${JOB_NAME}.config.json"
N_ATTEMPTS="${TBV21_PROTOCOLFIX_ATTEMPTS:-1}"
N_CONCURRENT="${TBV21_PROTOCOLFIX_CONCURRENCY:-1}"
MAX_RETRIES="${TBV21_PROTOCOLFIX_MAX_RETRIES:-0}"
TIMEOUT_MULTIPLIER="${TBV21_PROTOCOLFIX_TIMEOUT_MULTIPLIER:-1.0}"
DELETE_ENV="${TBV21_PROTOCOLFIX_DELETE_ENV:-1}"
export TBV21_ENABLE_SLIME_SAMPLING_FLAGS="${TBV21_ENABLE_SLIME_SAMPLING_FLAGS:-0}"
export TBV21_ENABLE_SAMPLING_SEED="${TBV21_ENABLE_SAMPLING_SEED:-0}"

TMUX_RUNTIME_DIR="${TBV21_TMUX_RUNTIME_DIR:-${TBV21_HOME}/runtime/tmux_runtime}"
APT_RUNTIME_DIR="${TBV21_APT_RUNTIME_DIR:-${TBV21_HOME}/runtime/apt}"
TOOLS_RUNTIME_DIR="${TBV21_TOOLS_RUNTIME_DIR:-${TBV21_HOME}/runtime/tools}"
WHEELHOUSE_DIR="${TBV21_WHEELHOUSE_DIR:-${TBV21_HOME}/runtime/wheelhouse}"
ADAPTER_DIR="${OPENCLAW_ADAPTER_DIR:-${TBV21_HOME}/runtime/openclaw_camel_adapter}"
CONTAINER_BASE_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
AGENT_CONTAINER_PATH="/opt/tbv21-tools/bin:/opt/tbv21-tmux/bin:${CONTAINER_BASE_PATH}"
ENV_CONTAINER_PATH="/opt/tbv21-tools/bin:${AGENT_CONTAINER_PATH}"

{
  printf 'export TBV21_PROTOCOLFIX_JOB_NAME=%q\n' "${JOB_NAME}"
  printf 'export TBV21_PROTOCOLFIX_LOG_FILE=%q\n' "${LOG_FILE}"
  printf 'export TBV21_PROTOCOLFIX_CONFIG_FILE=%q\n' "${CONFIG_FILE}"
  printf 'export TBV21_PROTOCOLFIX_TASKS=%q\n' "${TASKS[*]}"
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
log "enable_slime_sampling_flags=${TBV21_ENABLE_SLIME_SAMPLING_FLAGS}"
log "enable_sampling_seed=${TBV21_ENABLE_SAMPLING_SEED}"

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
for task in "${TASKS[@]}"; do
  if [ ! -d "${TBV21_TASKS_DIR}/${task}" ]; then
    log "[ERROR] task missing: ${TBV21_TASKS_DIR}/${task}"
    exit 1
  fi
done

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

TASKS_JSON="$(python3 - "$@" <<'PY'
import json, sys
tasks = sys.argv[1:] or ["hf-model-inference", "modernize-scientific-stack"]
print(json.dumps(tasks))
PY
)"

python3 - "${CONFIG_FILE}" "${JOB_NAME}" "${TASKS_JSON}" "${TMUX_RUNTIME_DIR}" "${APT_RUNTIME_DIR}" "${TOOLS_RUNTIME_DIR}" "${WHEELHOUSE_DIR}" "${ADAPTER_DIR}" "${ENV_CONTAINER_PATH}" "${AGENT_CONTAINER_PATH}" "${N_ATTEMPTS}" "${N_CONCURRENT}" "${MAX_RETRIES}" "${TIMEOUT_MULTIPLIER}" "${DELETE_ENV}" <<'PY'
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
) = sys.argv[1:16]

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

agent_env = {
    "OPENAI_API_KEY": "dummy",
    "OPENAI_BASE_URL": f"http://{os.environ['TBV21_SGLANG_HOST']}:{os.environ['TBV21_SGLANG_PORT']}/v1",
    "SGLANG_BASE_URL": f"http://{os.environ['TBV21_SGLANG_HOST']}:{os.environ['TBV21_SGLANG_PORT']}",
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
    "PYTHONPATH": adapter_dir,
    "TBV21_ENABLE_SLIME_SAMPLING_FLAGS": os.environ.get("TBV21_ENABLE_SLIME_SAMPLING_FLAGS", "0"),
    "TBV21_ENABLE_SAMPLING_SEED": os.environ.get("TBV21_ENABLE_SAMPLING_SEED", "0"),
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
                "sglang_base_url": f"http://{os.environ['TBV21_SGLANG_HOST']}:{os.environ['TBV21_SGLANG_PORT']}",
                "sglang_served_name": os.environ["TBV21_MODEL_NAME"],
                "hf_model_dir": os.environ["TBV21_MODEL_PATH"],
                "max_iteration": 10,
                "max_total_tokens": 16384,
                "max_input_tokens": 8192,
                "max_new_tokens": 8192,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "skip_special_tokens": False,
                "eval_seed": 42,
                "enable_slime_sampling_flags": os.environ.get("TBV21_ENABLE_SLIME_SAMPLING_FLAGS", "0") == "1",
                "enable_sampling_seed": os.environ.get("TBV21_ENABLE_SAMPLING_SEED", "0") == "1",
                "sampling_seed": 42,
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

log "validating adapter import and generated config"
PYTHONPATH="${ADAPTER_DIR}:${PYTHONPATH:-}" /mnt/shared-storage-user/tangjia/miniconda3/envs/terminal-bench/bin/python - <<'PY'
import importlib
mod = importlib.import_module("openclaw_camel_adapter")
cls = mod.OpenClawCamelAgent
print("[OK] adapter", cls.name(), cls.version(cls.__new__(cls)))
PY
python3 -m json.tool "${CONFIG_FILE}" >/dev/null

log "starting Harbor protocolfix subset eval"
PYTHONPATH="${ADAPTER_DIR}:${PYTHONPATH:-}" "${TBV21_HARBOR_BIN}" run --config "${CONFIG_FILE}"

log "Harbor protocolfix subset eval completed"
"${SCRIPT_DIR}/summarize_harbor_job.py" "${TBV21_JOBS_DIR}/${JOB_NAME}" || true
