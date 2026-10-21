#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 /path/to/upstream_gist_job.env" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TBV21_HOME="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${TBV21_HOME}/env.sh"
# shellcheck disable=SC1090
source "$1"

mkdir -p "${TBV21_LOG_DIR}" "${TBV21_STATE_DIR}" "${TBV21_JOBS_DIR}"
touch "${TBV21_UPSTREAM_GIST_LOG_FILE}" "${TBV21_UPSTREAM_GIST_BOOTSTRAP_LOG}"
exec > >(tee -a "${TBV21_UPSTREAM_GIST_BOOTSTRAP_LOG}" "${TBV21_UPSTREAM_GIST_LOG_FILE}") 2>&1

log() {
  echo "[$(date '+%F %T')] $*"
}

log "job=${TBV21_UPSTREAM_GIST_JOB_NAME}"
log "log=${TBV21_UPSTREAM_GIST_LOG_FILE}"
log "config=${TBV21_UPSTREAM_GIST_CONFIG_FILE}"
log "model=${TBV21_MODEL_NAME}"
log "model_path=${TBV21_MODEL_PATH}"
log "ckpt=${TBV21_CKPT_PATH:-}"
log "tasks=$(wc -w <<< "${TBV21_UPSTREAM_GIST_TASKS:-}") attempts/concurrency come from config"

if [ ! -f "${TBV21_UPSTREAM_GIST_CONFIG_FILE}" ]; then
  log "[ERROR] config not found: ${TBV21_UPSTREAM_GIST_CONFIG_FILE}"
  exit 1
fi
if [ ! -d "${TBV21_MODEL_PATH}" ]; then
  log "[ERROR] model path not found: ${TBV21_MODEL_PATH}"
  exit 1
fi

log "adapter/source hashes"
sha256sum \
  "${TBV21_HOME}/runtime/openclaw_camel_adapter_upstream_gist/openclaw_camel_adapter.py" \
  "${REAL_OPENCLAW_TERMINAL_RL_DIR}/inference_client.py" \
  "${REAL_OPENCLAW_TERMINAL_RL_DIR}/agent_runner.py"

log "starting/checking rootless Docker backend"
"${SCRIPT_DIR}/use_worker_rootless_docker.sh" start
# shellcheck disable=SC1090
source "${TBV21_DOCKER_HOST_ENV}"
docker info --format 'server={{.ServerVersion}} root={{.DockerRootDir}} driver={{.Driver}} cgroup={{.CgroupDriver}} containers={{.Containers}} images={{.Images}}'

log "checking SGLang endpoint"
curl --noproxy '*' -fsS --max-time 10 "http://${TBV21_SGLANG_HOST}:${TBV21_SGLANG_PORT}/v1/models"
echo

log "validating upstream gist adapter import and generated config"
"${TBV21_HARBOR_PYTHON:-/mnt/shared-storage-user/tangjia/miniconda3/envs/terminal-bench/bin/python}" - <<'PY'
import importlib
import json
import os
from pathlib import Path

cfg_path = Path(os.environ["TBV21_UPSTREAM_GIST_CONFIG_FILE"])
cfg = json.loads(cfg_path.read_text())
agent = cfg["agents"][0]
mod_name, cls_name = agent["import_path"].split(":")
mod = importlib.import_module(mod_name)
cls = getattr(mod, cls_name)
print("[OK] adapter", cls.__name__)
print("[OK] import_path", agent["import_path"])
print("[OK] model_name", agent["model_name"])
print("[OK] served", agent["kwargs"]["sglang_served_name"])
print("[OK] hf_model_dir", agent["kwargs"]["hf_model_dir"])
print("[OK] attempts", cfg["n_attempts"], "concurrency", cfg["n_concurrent_trials"])
print("[OK] tasks", len(cfg["datasets"][0].get("task_names") or []))
PY

log "starting Harbor upstream gist adapter eval"
bin/harbor run -c "${TBV21_UPSTREAM_GIST_CONFIG_FILE}" -y

log "Harbor upstream gist adapter eval completed"
python3 "${SCRIPT_DIR}/analyze_tbv21_harbor_job.py" \
  "${TBV21_JOBS_DIR}/${TBV21_UPSTREAM_GIST_JOB_NAME}" \
  --print-summary
