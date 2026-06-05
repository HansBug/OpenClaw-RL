#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-user/puyuan/code/OpenClaw-RL}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REPO_ROOT}/tmp_doc_latest/remote_logs}"
CPU_WORKER_ID="${CPU_WORKER_ID:-$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo unknown-worker)}"
CPU_WORKER_ID="$(printf '%s' "${CPU_WORKER_ID}" | tr -c 'A-Za-z0-9_.-' '_')"
OPENCLAW_REMOTE_RUN_ID="${OPENCLAW_REMOTE_RUN_ID:-$(date +%Y%m%d_%H%M%S)_pid$$}"
OPENCLAW_REMOTE_LOG_DIR="${OPENCLAW_REMOTE_LOG_DIR:-${REMOTE_LOG_ROOT}/${CPU_WORKER_ID}/${OPENCLAW_REMOTE_RUN_ID}}"
export REPO_ROOT REMOTE_LOG_ROOT CPU_WORKER_ID OPENCLAW_REMOTE_RUN_ID OPENCLAW_REMOTE_LOG_DIR
export LOG_FILE="${LOG_FILE:-${OPENCLAW_REMOTE_LOG_DIR}/docker_watchdog.log}"
export POOL_HOST="${POOL_HOST:-127.0.0.1}"
export POOL_PORT="${POOL_PORT:-18081}"
export POOL_CHECK_INTERVAL="${POOL_CHECK_INTERVAL:-30}"
export POOL_PENDING_CLOSES_WARN="${POOL_PENDING_CLOSES_WARN:-50}"
export POOL_PENDING_CLOSES_REPAIR="${POOL_PENDING_CLOSES_REPAIR:-1}"
export POOL_PENDING_CLOSES_REPAIR_THRESHOLD="${POOL_PENDING_CLOSES_REPAIR_THRESHOLD:-50}"
export POOL_PENDING_CLOSES_STUCK_CHECKS="${POOL_PENDING_CLOSES_STUCK_CHECKS:-2}"
export POOL_PENDING_CLOSES_ACTIVE_MAX="${POOL_PENDING_CLOSES_ACTIVE_MAX:-64}"
export POOL_PENDING_CLOSES_REAP_LIMIT="${POOL_PENDING_CLOSES_REAP_LIMIT:-0}"
export POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S="${POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S:-120}"
export POOL_PENDING_CLOSES_CANCEL_API="${POOL_PENDING_CLOSES_CANCEL_API:-1}"
export POOL_PENDING_CLOSES_CANCEL_TIMEOUT="${POOL_PENDING_CLOSES_CANCEL_TIMEOUT:-5}"
export POOL_PENDING_CLOSES_CANCEL_MIN_AGE="${POOL_PENDING_CLOSES_CANCEL_MIN_AGE:-45}"
export MAX_RUNNING_CONTAINERS="${MAX_RUNNING_CONTAINERS:-96}"
export HARD_KILL_THRESHOLD="${HARD_KILL_THRESHOLD:-160}"
export TASK_CONTAINER_REGEX='^[0-9]+[-_].*(slime[-_]?run|client|helper).*$'
export TASK_IMAGE_REGEX='^tb__[0-9]+__.*(:|$)'
export WATCHDOG_AGGRESSIVE_IMAGE_PRUNE="${WATCHDOG_AGGRESSIVE_IMAGE_PRUNE:-0}"

mkdir -p "$(dirname "${LOG_FILE}")"
touch "${LOG_FILE}"
ln -sfn "${OPENCLAW_REMOTE_LOG_DIR}" "${REMOTE_LOG_ROOT}/${CPU_WORKER_ID}/latest_watchdog" 2>/dev/null || true

cd "${REPO_ROOT}"
{
    echo "========================================"
    echo "  OpenClaw docker watchdog"
    echo "  repo:       ${REPO_ROOT}"
    echo "  worker_id:  ${CPU_WORKER_ID}"
    echo "  run_id:     ${OPENCLAW_REMOTE_RUN_ID}"
    echo "  log_dir:    ${OPENCLAW_REMOTE_LOG_DIR}"
    echo "  log_file:   ${LOG_FILE}"
    echo "  pool:       ${POOL_HOST}:${POOL_PORT}"
    echo "========================================"
} | tee -a "${LOG_FILE}"
exec bash terminal-rl/remote/docker_watchdog_v2.sh 2>&1 | tee -a "${LOG_FILE}"
