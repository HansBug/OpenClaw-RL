#!/usr/bin/env bash
# CPU-worker pool_server launcher for SetA/terminal-rl.
#
# Runs in the foreground. Start it inside tmux/screen or redirect it from the
# caller if you want it detached.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-user/puyuan/code/OpenClaw-RL}"
cd "${REPO_ROOT}"

RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
if [[ -n "${RUN_DIR:-}" ]]; then
    DEFAULT_REMOTE_LOG_ROOT="${RUN_DIR}/remote_logs"
elif [[ -n "${RUN_ID:-}" ]]; then
    DEFAULT_REMOTE_LOG_ROOT="${RUNS_ROOT}/${RUN_ID}/remote_logs"
else
    DEFAULT_REMOTE_LOG_ROOT="${RUNS_ROOT}/remote_logs"
fi
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${DEFAULT_REMOTE_LOG_ROOT}}"
CPU_WORKER_ID="${CPU_WORKER_ID:-$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo unknown-worker)}"
CPU_WORKER_ID="$(printf '%s' "${CPU_WORKER_ID}" | tr -c 'A-Za-z0-9_.-' '_')"
OPENCLAW_REMOTE_RUN_ID="${OPENCLAW_REMOTE_RUN_ID:-$(date +%Y%m%d_%H%M%S)_pid$$}"
OPENCLAW_REMOTE_LOG_DIR="${OPENCLAW_REMOTE_LOG_DIR:-${REMOTE_LOG_ROOT}/${CPU_WORKER_ID}/${OPENCLAW_REMOTE_RUN_ID}}"
export RUNS_ROOT REMOTE_LOG_ROOT CPU_WORKER_ID OPENCLAW_REMOTE_RUN_ID OPENCLAW_REMOTE_LOG_DIR
export CPU_POOL_LOG="${CPU_POOL_LOG:-${OPENCLAW_REMOTE_LOG_DIR}/cpu_pool.log}"
export CPU_ERR_LOG="${CPU_ERR_LOG:-${OPENCLAW_REMOTE_LOG_DIR}/cpu_err.log}"
mkdir -p "${OPENCLAW_REMOTE_LOG_DIR}" "${REMOTE_LOG_ROOT}/${CPU_WORKER_ID}"
ln -sfnT "${OPENCLAW_REMOTE_LOG_DIR}" "${REMOTE_LOG_ROOT}/${CPU_WORKER_ID}/latest_server" 2>/dev/null || true

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# Capacity. Keep default Docker concurrency near one 8B GRPO demand
# (rollout-batch-size 16 x n-samples 8 = 128). Raise explicitly only after
# pids headroom is proven on the worker.
export WORKER_MAX_TASKS="${WORKER_MAX_TASKS:-16}"
export WORKER_MAX_RUNS_PER_TASK="${WORKER_MAX_RUNS_PER_TASK:-8}"
export WORKER_MAX_CONCURRENT_CLOSES="${WORKER_MAX_CONCURRENT_CLOSES:-16}"
export WORKER_MAX_CONCURRENT_BUILDS="${WORKER_MAX_CONCURRENT_BUILDS:-2}"
export WORKER_RUN_IDLE_TTL="${WORKER_RUN_IDLE_TTL:-180}"
export ENV_SERVER_PORT="${ENV_SERVER_PORT:-18081}"

# Timeouts and admission guards.
# P0 fix: Increase reset timeout tolerance for Docker operations under load
export WORKER_CLOSE_TASK_TIMEOUT="${WORKER_CLOSE_TASK_TIMEOUT:-45}"
export WORKER_ALLOCATED_TTL="${WORKER_ALLOCATED_TTL:-60}"  # 120→60s: CRITICAL fix for slot accumulation
export WORKER_RESET_OPERATION_TIMEOUT="${WORKER_RESET_OPERATION_TIMEOUT:-720}"  # 330→720s: allow 12min for Docker ops under pressure
export WORKER_RESETTING_TTL="${WORKER_RESETTING_TTL:-900}"  # 390→900s: match increased reset timeout
export WORKER_CLOSING_REQUESTED_TTL="${WORKER_CLOSING_REQUESTED_TTL:-300}"
export WORKER_PIDS_PAUSE_ALLOCATE_PCT="${WORKER_PIDS_PAUSE_ALLOCATE_PCT:-80}"
export WORKER_PIDS_REJECT_RESET_PCT="${WORKER_PIDS_REJECT_RESET_PCT:-85}"
export WORKER_PIDS_MIN_FREE_ALLOCATE="${WORKER_PIDS_MIN_FREE_ALLOCATE:-6000}"
export WORKER_PIDS_MIN_FREE_RESET="${WORKER_PIDS_MIN_FREE_RESET:-4000}"
# P0 fix: Lower shim thresholds to trigger proactive cleanup earlier
export WORKER_SHIM_PAUSE_ALLOCATE="${WORKER_SHIM_PAUSE_ALLOCATE:-160}"  # 180→160: cleanup before hitting critical
export WORKER_SHIM_REJECT_RESET="${WORKER_SHIM_REJECT_RESET:-200}"  # 220→200: more headroom
export WORKER_PENDING_CLOSES_PAUSE_ALLOCATE="${WORKER_PENDING_CLOSES_PAUSE_ALLOCATE:-32}"
export WORKER_PENDING_CLOSES_REJECT_RESET="${WORKER_PENDING_CLOSES_REJECT_RESET:-64}"
export WORKER_DOCKER_CLI_TIMEOUT="${WORKER_DOCKER_CLI_TIMEOUT:-3}"
export WORKER_PRESSURE_CACHE_TTL="${WORKER_PRESSURE_CACHE_TTL:-5}"
export CONTAINER_PIDS_LIMIT="${CONTAINER_PIDS_LIMIT:-64}"
export CONTAINER_MEMORY_LIMIT="${CONTAINER_MEMORY_LIMIT:-16g}"

# Preflight and Docker cleanup. This worker is assumed dedicated to OpenClaw
# experiments, so broad cleanup is allowed to remove stale task containers.
export PREFLIGHT_KILL_ORPHAN_RUNNING="${PREFLIGHT_KILL_ORPHAN_RUNNING:-1}"
export SKIP_PREFLIGHT_CLEANUP="${SKIP_PREFLIGHT_CLEANUP:-0}"
export FINAL_DOCKER_CLEANUP="${FINAL_DOCKER_CLEANUP:-1}"
export FINAL_DOCKER_CLEANUP_TIMEOUT="${FINAL_DOCKER_CLEANUP_TIMEOUT:-90}"
export POOL_SERVER_SHUTDOWN_GRACE="${POOL_SERVER_SHUTDOWN_GRACE:-60}"
export TERMINAL_ENV_FORCE_DOCKER_CLEANUP="${TERMINAL_ENV_FORCE_DOCKER_CLEANUP:-1}"
export TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD="${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD:-1}"
export TERMINAL_ENV_FORCE_DOCKER_CLEANUP_ALWAYS="${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_ALWAYS:-1}"
export TERMINAL_ENV_FORCE_DOCKER_CLEANUP_TIMEOUT="${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_TIMEOUT:-20}"
export TERMINAL_ENV_FAST_CLOSE="${TERMINAL_ENV_FAST_CLOSE:-1}"
export TERMINAL_ENV_SKIP_UNBOUNDED_STOP="${TERMINAL_ENV_SKIP_UNBOUNDED_STOP:-1}"
export TERMINAL_ENV_FAST_CLOSE_STOP_TIMEOUT="${TERMINAL_ENV_FAST_CLOSE_STOP_TIMEOUT:-5}"

# Repair stale in-memory pending close tasks. These are not Docker containers;
# they are asyncio close tasks inside pool_server.
export WORKER_REPAIR_PENDING_CLOSES="${WORKER_REPAIR_PENDING_CLOSES:-1}"
export WORKER_REPAIR_PENDING_CLOSES_MAX_ACTIVE_RUNS="${WORKER_REPAIR_PENDING_CLOSES_MAX_ACTIVE_RUNS:--1}"
export WORKER_REPAIR_PENDING_CLOSES_CANCEL_TIMEOUT="${WORKER_REPAIR_PENDING_CLOSES_CANCEL_TIMEOUT:-5}"
export WORKER_REPAIR_PENDING_CLOSES_MIN_AGE="${WORKER_REPAIR_PENDING_CLOSES_MIN_AGE:-45}"
export WORKER_REPAIR_STALE_RUNS="${WORKER_REPAIR_STALE_RUNS:-1}"
export WORKER_REPAIR_STALE_RUNS_MIN_AGE="${WORKER_REPAIR_STALE_RUNS_MIN_AGE:-0}"
export WORKER_REPAIR_STALE_RUNS_MAX_REPAIRS="${WORKER_REPAIR_STALE_RUNS_MAX_REPAIRS:-20}"
export WORKER_REPAIR_CLOSE_REQUESTED_RUNS="${WORKER_REPAIR_CLOSE_REQUESTED_RUNS:-1}"
export WORKER_REPAIR_CLOSE_REQUESTED_MIN_AGE="${WORKER_REPAIR_CLOSE_REQUESTED_MIN_AGE:-0}"
export WORKER_REPAIR_CLOSE_REQUESTED_MAX_REPAIRS="${WORKER_REPAIR_CLOSE_REQUESTED_MAX_REPAIRS:-20}"
export WORKER_REPAIR_RESETTING_RUNS="${WORKER_REPAIR_RESETTING_RUNS:-1}"
export WORKER_REPAIR_RESETTING_MIN_AGE="${WORKER_REPAIR_RESETTING_MIN_AGE:-900}"  # 390→900s: match new RESETTING_TTL
export WORKER_REPAIR_RESETTING_MAX_REPAIRS="${WORKER_REPAIR_RESETTING_MAX_REPAIRS:-64}"
export WORKER_CLOSE_REQUESTED_FORCE_RELEASE="${WORKER_CLOSE_REQUESTED_FORCE_RELEASE:-1}"
export WORKER_CLOSE_REQUESTED_FORCE_RELEASE_AFTER="${WORKER_CLOSE_REQUESTED_FORCE_RELEASE_AFTER:-30}"
export WORKER_AUTO_REPAIR_ON_CAPACITY="${WORKER_AUTO_REPAIR_ON_CAPACITY:-1}"
export WORKER_AUTO_REPAIR_CLOSE_REQUESTED_MIN_AGE="${WORKER_AUTO_REPAIR_CLOSE_REQUESTED_MIN_AGE:-0}"
export WORKER_AUTO_REPAIR_STALE_MIN_AGE="${WORKER_AUTO_REPAIR_STALE_MIN_AGE:-0}"
export WORKER_AUTO_REPAIR_MAX_REPAIRS="${WORKER_AUTO_REPAIR_MAX_REPAIRS:-40}"  # 20→40: aggressive cleanup during exhaustion

# Avoid duplicate concurrent builds for the same task image.
export WORKER_DOCKER_BUILD_DEDUP="${WORKER_DOCKER_BUILD_DEDUP:-1}"
export WORKER_DOCKER_BUILD_SKIP_EXISTING="${WORKER_DOCKER_BUILD_SKIP_EXISTING:-1}"
export WORKER_DOCKER_BUILD_FAILED_TTL="${WORKER_DOCKER_BUILD_FAILED_TTL:-3600}"

# ClawSentry is not needed for pure SetA outcome-reward baselines.
export CLAWSENTRY_NEEDED="${CLAWSENTRY_NEEDED:-0}"

echo "========================================"
echo "  OpenClaw pool_server"
echo "  repo:                 ${REPO_ROOT}"
echo "  python:               $(command -v python || true)"
echo "  port:                 ${ENV_SERVER_PORT}"
echo "  worker_id:            ${CPU_WORKER_ID}"
echo "  run_id:               ${OPENCLAW_REMOTE_RUN_ID}"
echo "  log_dir:              ${OPENCLAW_REMOTE_LOG_DIR}"
echo "  server_log:           ${CPU_POOL_LOG}"
echo "  err_log:              ${CPU_ERR_LOG}"
echo "  max_tasks:            ${WORKER_MAX_TASKS}"
echo "  max_runs_per_task:    ${WORKER_MAX_RUNS_PER_TASK}"
echo "  concurrent_closes:    ${WORKER_MAX_CONCURRENT_CLOSES}"
echo "  concurrent_builds:    ${WORKER_MAX_CONCURRENT_BUILDS}"
echo "  idle_ttl:             ${WORKER_RUN_IDLE_TTL}s"
echo "  close_timeout:        ${WORKER_CLOSE_TASK_TIMEOUT}s"
echo "  reset_timeout:        operation=${WORKER_RESET_OPERATION_TIMEOUT}s stale_ttl=${WORKER_RESETTING_TTL}s"
echo "  stale_ttl:            allocated=${WORKER_ALLOCATED_TTL}s resetting=${WORKER_RESETTING_TTL}s closing_requested=${WORKER_CLOSING_REQUESTED_TTL}s"
echo "  pids_guard:           allocate=${WORKER_PIDS_PAUSE_ALLOCATE_PCT}%/${WORKER_PIDS_MIN_FREE_ALLOCATE}free reset=${WORKER_PIDS_REJECT_RESET_PCT}%/${WORKER_PIDS_MIN_FREE_RESET}free"
echo "  shim_guard:           allocate=${WORKER_SHIM_PAUSE_ALLOCATE} reset=${WORKER_SHIM_REJECT_RESET}"
echo "  pending_close_guard:  allocate=${WORKER_PENDING_CLOSES_PAUSE_ALLOCATE} reset=${WORKER_PENDING_CLOSES_REJECT_RESET}"
echo "  container_limits:     pids=${CONTAINER_PIDS_LIMIT} memory=${CONTAINER_MEMORY_LIMIT}"
echo "  preflight_cleanup:    skip=${SKIP_PREFLIGHT_CLEANUP} kill_orphans=${PREFLIGHT_KILL_ORPHAN_RUNNING}"
echo "  final_cleanup:        enabled=${FINAL_DOCKER_CLEANUP} timeout=${FINAL_DOCKER_CLEANUP_TIMEOUT}s shutdown_grace=${POOL_SERVER_SHUTDOWN_GRACE}s"
echo "  force_cleanup:        enabled=${TERMINAL_ENV_FORCE_DOCKER_CLEANUP} broad=${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD} always=${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_ALWAYS}"
echo "  fast_close:           enabled=${TERMINAL_ENV_FAST_CLOSE} skip_unbounded_stop=${TERMINAL_ENV_SKIP_UNBOUNDED_STOP} stop_timeout=${TERMINAL_ENV_FAST_CLOSE_STOP_TIMEOUT}s"
echo "  pending_repair:       enabled=${WORKER_REPAIR_PENDING_CLOSES} max_active=${WORKER_REPAIR_PENDING_CLOSES_MAX_ACTIVE_RUNS} min_age=${WORKER_REPAIR_PENDING_CLOSES_MIN_AGE}s"
echo "  stale_run_repair:     enabled=${WORKER_REPAIR_STALE_RUNS} min_age=${WORKER_REPAIR_STALE_RUNS_MIN_AGE}s max_repairs=${WORKER_REPAIR_STALE_RUNS_MAX_REPAIRS}"
echo "  close_req_repair:     enabled=${WORKER_REPAIR_CLOSE_REQUESTED_RUNS} min_age=${WORKER_REPAIR_CLOSE_REQUESTED_MIN_AGE}s max_repairs=${WORKER_REPAIR_CLOSE_REQUESTED_MAX_REPAIRS}"
echo "  resetting_repair:     enabled=${WORKER_REPAIR_RESETTING_RUNS} min_age=${WORKER_REPAIR_RESETTING_MIN_AGE}s max_repairs=${WORKER_REPAIR_RESETTING_MAX_REPAIRS}"
echo "  close_req_release:    enabled=${WORKER_CLOSE_REQUESTED_FORCE_RELEASE} after=${WORKER_CLOSE_REQUESTED_FORCE_RELEASE_AFTER}s"
echo "  capacity_auto_repair: enabled=${WORKER_AUTO_REPAIR_ON_CAPACITY} close_min_age=${WORKER_AUTO_REPAIR_CLOSE_REQUESTED_MIN_AGE}s stale_min_age=${WORKER_AUTO_REPAIR_STALE_MIN_AGE}s max_repairs=${WORKER_AUTO_REPAIR_MAX_REPAIRS}"
echo "  build_dedup:          enabled=${WORKER_DOCKER_BUILD_DEDUP} skip_existing=${WORKER_DOCKER_BUILD_SKIP_EXISTING} failed_ttl=${WORKER_DOCKER_BUILD_FAILED_TTL}s"
echo "========================================"

exec bash terminal-rl/remote/run_pool_server_pu_v2.sh
