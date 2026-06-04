#!/usr/bin/env bash
# CPU-worker pool_server launcher for SetA/terminal-rl.
#
# Runs in the foreground. Start it inside tmux/screen or redirect it from the
# caller if you want it detached.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-user/puyuan/code/OpenClaw-RL}"
cd "${REPO_ROOT}"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# Capacity. For one shared CPU worker serving 1-2 GPU jobs, this keeps enough
# leases while avoiding excessive Docker build/close fan-out.
export WORKER_MAX_TASKS="${WORKER_MAX_TASKS:-32}"
export WORKER_MAX_RUNS_PER_TASK="${WORKER_MAX_RUNS_PER_TASK:-8}"
export WORKER_MAX_CONCURRENT_CLOSES="${WORKER_MAX_CONCURRENT_CLOSES:-32}"
export WORKER_MAX_CONCURRENT_BUILDS="${WORKER_MAX_CONCURRENT_BUILDS:-4}"
export WORKER_RUN_IDLE_TTL="${WORKER_RUN_IDLE_TTL:-180}"
export ENV_SERVER_PORT="${ENV_SERVER_PORT:-18081}"

# Timeouts and admission guards.
export WORKER_CLOSE_TASK_TIMEOUT="${WORKER_CLOSE_TASK_TIMEOUT:-90}"
export WORKER_PENDING_CLOSES_PAUSE_ALLOCATE="${WORKER_PENDING_CLOSES_PAUSE_ALLOCATE:-64}"
export WORKER_PENDING_CLOSES_REJECT_RESET="${WORKER_PENDING_CLOSES_REJECT_RESET:-128}"
export WORKER_DOCKER_CLI_TIMEOUT="${WORKER_DOCKER_CLI_TIMEOUT:-3}"
export WORKER_PRESSURE_CACHE_TTL="${WORKER_PRESSURE_CACHE_TTL:-5}"

# Preflight and Docker cleanup. This worker is assumed dedicated to OpenClaw
# experiments, so broad cleanup is allowed to remove stale task containers.
export PREFLIGHT_KILL_ORPHAN_RUNNING="${PREFLIGHT_KILL_ORPHAN_RUNNING:-1}"
export SKIP_PREFLIGHT_CLEANUP="${SKIP_PREFLIGHT_CLEANUP:-0}"
export TERMINAL_ENV_FORCE_DOCKER_CLEANUP="${TERMINAL_ENV_FORCE_DOCKER_CLEANUP:-1}"
export TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD="${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD:-1}"
export TERMINAL_ENV_FORCE_DOCKER_CLEANUP_ALWAYS="${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_ALWAYS:-1}"
export TERMINAL_ENV_FORCE_DOCKER_CLEANUP_TIMEOUT="${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_TIMEOUT:-20}"

# Repair stale in-memory pending close tasks. These are not Docker containers;
# they are asyncio close tasks inside pool_server.
export WORKER_REPAIR_PENDING_CLOSES="${WORKER_REPAIR_PENDING_CLOSES:-1}"
export WORKER_REPAIR_PENDING_CLOSES_MAX_ACTIVE_RUNS="${WORKER_REPAIR_PENDING_CLOSES_MAX_ACTIVE_RUNS:-64}"
export WORKER_REPAIR_PENDING_CLOSES_CANCEL_TIMEOUT="${WORKER_REPAIR_PENDING_CLOSES_CANCEL_TIMEOUT:-5}"
export WORKER_REPAIR_PENDING_CLOSES_MIN_AGE="${WORKER_REPAIR_PENDING_CLOSES_MIN_AGE:-90}"

# Avoid duplicate concurrent builds for the same task image.
export WORKER_DOCKER_BUILD_DEDUP="${WORKER_DOCKER_BUILD_DEDUP:-1}"
export WORKER_DOCKER_BUILD_SKIP_EXISTING="${WORKER_DOCKER_BUILD_SKIP_EXISTING:-1}"

# ClawSentry is not needed for pure SetA outcome-reward baselines.
export CLAWSENTRY_NEEDED="${CLAWSENTRY_NEEDED:-0}"

echo "========================================"
echo "  OpenClaw pool_server"
echo "  repo:                 ${REPO_ROOT}"
echo "  python:               $(command -v python || true)"
echo "  port:                 ${ENV_SERVER_PORT}"
echo "  max_tasks:            ${WORKER_MAX_TASKS}"
echo "  max_runs_per_task:    ${WORKER_MAX_RUNS_PER_TASK}"
echo "  concurrent_closes:    ${WORKER_MAX_CONCURRENT_CLOSES}"
echo "  concurrent_builds:    ${WORKER_MAX_CONCURRENT_BUILDS}"
echo "  idle_ttl:             ${WORKER_RUN_IDLE_TTL}s"
echo "  pending_close_guard:  allocate=${WORKER_PENDING_CLOSES_PAUSE_ALLOCATE} reset=${WORKER_PENDING_CLOSES_REJECT_RESET}"
echo "  force_cleanup:        enabled=${TERMINAL_ENV_FORCE_DOCKER_CLEANUP} broad=${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD} always=${TERMINAL_ENV_FORCE_DOCKER_CLEANUP_ALWAYS}"
echo "  pending_repair:       enabled=${WORKER_REPAIR_PENDING_CLOSES} max_active=${WORKER_REPAIR_PENDING_CLOSES_MAX_ACTIVE_RUNS} min_age=${WORKER_REPAIR_PENDING_CLOSES_MIN_AGE}s"
echo "  build_dedup:          enabled=${WORKER_DOCKER_BUILD_DEDUP} skip_existing=${WORKER_DOCKER_BUILD_SKIP_EXISTING}"
echo "========================================"

exec bash terminal-rl/remote/run_pool_server_pu_v2.sh
