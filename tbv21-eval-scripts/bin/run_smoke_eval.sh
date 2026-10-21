#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../env.sh"

TASK="${1:-overfull-hbox}"
JOB_NAME="${2:-smoke_v21_${TBV21_MODEL_NAME}_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${TBV21_LOG_DIR}/${JOB_NAME}.log"

if [ ! -d "${TBV21_TASKS_DIR}" ]; then
  echo "[ERROR] tasks dir missing: ${TBV21_TASKS_DIR}" >&2
  exit 1
fi

if docker info >/dev/null 2>&1 && docker run --rm hello-world >/dev/null 2>&1; then
  echo "[OK] Docker backend already usable: ${DOCKER_HOST}"
elif [ -x "${SCRIPT_DIR}/use_worker_rootless_docker.sh" ]; then
  "${SCRIPT_DIR}/use_worker_rootless_docker.sh" smoke
else
  "${SCRIPT_DIR}/start_docker.sh" >/dev/null
fi

if ! curl --noproxy '*' -fsS --max-time 5 "http://${TBV21_SGLANG_HOST}:${TBV21_SGLANG_PORT}/v1/models" >/dev/null; then
  echo "[ERROR] sglang endpoint is not ready: http://${TBV21_SGLANG_HOST}:${TBV21_SGLANG_PORT}/v1" >&2
  exit 1
fi

echo "[INFO] running TB v2.1 smoke eval task=${TASK} job=${JOB_NAME}"
echo "[INFO] log=${LOG_FILE}"

set -x
"${TBV21_HARBOR_BIN}" run \
  -p "${TBV21_TASKS_DIR}" \
  --agent terminus-2 \
  --model "openai/${TBV21_MODEL_NAME}" \
  --ak "api_base=http://${TBV21_SGLANG_HOST}:${TBV21_SGLANG_PORT}/v1" \
  --ae "OPENAI_API_KEY=dummy" \
  --ae "OPENAI_BASE_URL=http://${TBV21_SGLANG_HOST}:${TBV21_SGLANG_PORT}/v1" \
  -e docker \
  -o "${TBV21_JOBS_DIR}" \
  --job-name "${JOB_NAME}" \
  -i "${TASK}" \
  -k 1 \
  -n 1 \
  2>&1 | tee "${LOG_FILE}"
set +x

echo "[OK] smoke eval command completed"
find "${TBV21_JOBS_DIR}/${JOB_NAME}" -maxdepth 4 -type f | sort | sed -n '1,120p' || true
