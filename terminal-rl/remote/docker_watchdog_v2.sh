#!/usr/bin/env bash
# docker_watchdog_v2.sh — 加固版守护进程（针对 DinD/DooD + agentic-RL 场景）
#
# 设计目标：在 14 h 长跑中拦截 issue #3 §3/§4/§5 描述的环境层崩溃
#   §3 docker bridge address-pool 耗尽
#   §4 nofile=1024 触发 "Too many open files"
#   §5 长跑后 dockerd 状态污染（孤儿容器/网络残留）
#
# 修复了 v2 早期版本的 9 个问题（详见 ../runs/.../analysis/REPORT.md → docker_watchdog 修复方案）：
#   P0-1: 删除 restart_docker 中的 systemctl restart 调用（D state 时会无限挂起）
#   P0-2: pkill 前先 stop docker.socket，阻断 systemd auto-restart race
#   P0-3: emergency_pressure_relief 加 60 s 冷却 + foreground+timeout，避免并发拖死 dockerd
#   P1-1: 新增 pool_server /healthz + /status + bridge 网络数 监控（这次崩溃的真正信号）
#   P1-2: cgroup v2 detection（统一 hierarchy 路径解析）
#   P1-3: 检测 host pid namespace；不在 host ns 时不擦容器 state
#   P1-4: 周期性深度探活（network create/rm 模拟 pool 真实路径）
#   P1-5: 日志 rotate 改为 truncate-in-place，nohup fd 不丢
#   P1-6: enforce_container_limit 排除 pool_server 容器（杀候选改用 task 容器 pattern）
#   P1-7: 启动时打印 namespace 信息便于诊断
#
# 用法（推荐 systemd 起，见 docker-watchdog.service）：
#   systemctl enable --now docker-watchdog
# 或临时：
#   nohup bash docker_watchdog_v2.sh > /tmp/docker_watchdog.log 2>&1 &

set -uo pipefail

# ── 可调参数 ──────────────────────────────────────────────────────────
MAX_RUNNING_CONTAINERS="${MAX_RUNNING_CONTAINERS:-80}"
HARD_KILL_THRESHOLD="${HARD_KILL_THRESHOLD:-120}"
CLEANUP_INTERVAL="${CLEANUP_INTERVAL:-60}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-30}"
CGROUP_MONITOR_INTERVAL="${CGROUP_MONITOR_INTERVAL:-15}"
PROC_MONITOR_INTERVAL="${PROC_MONITOR_INTERVAL:-15}"
DOCKER_CLI_CHECK_INTERVAL="${DOCKER_CLI_CHECK_INTERVAL:-30}"
PROXY_CHECK_INTERVAL="${PROXY_CHECK_INTERVAL:-300}"
POOL_CHECK_INTERVAL="${POOL_CHECK_INTERVAL:-30}"
DEEP_PROBE_INTERVAL="${DEEP_PROBE_INTERVAL:-300}"
PIDS_WARN_PCT="${PIDS_WARN_PCT:-75}"
PIDS_EMERGENCY_PCT="${PIDS_EMERGENCY_PCT:-90}"
PROC_WARN_COOLDOWN_S="${PROC_WARN_COOLDOWN_S:-60}"
PIDS_RELIEF_COOLDOWN_S="${PIDS_RELIEF_COOLDOWN_S:-30}"
DOCKER_PROC_WARN="${DOCKER_PROC_WARN:-512}"
DOCKER_PROC_EMERGENCY="${DOCKER_PROC_EMERGENCY:-900}"
SHIM_PROC_WARN="${SHIM_PROC_WARN:-256}"
SHIM_PROC_EMERGENCY="${SHIM_PROC_EMERGENCY:-512}"
DOCKER_DOWN_SHIM_RELIEF="${DOCKER_DOWN_SHIM_RELIEF:-128}"
RUNC_PROC_WARN="${RUNC_PROC_WARN:-50}"
RUNC_PROC_EMERGENCY="${RUNC_PROC_EMERGENCY:-150}"
ZOMBIE_WARN="${ZOMBIE_WARN:-50}"
ZOMBIE_EMERGENCY="${ZOMBIE_EMERGENCY:-200}"
MEM_WARN_PCT="${MEM_WARN_PCT:-80}"
MEM_EMERGENCY_PCT="${MEM_EMERGENCY_PCT:-92}"
MAX_CONSECUTIVE_HEALTH_FAILS="${MAX_CONSECUTIVE_HEALTH_FAILS:-3}"
MAX_CONSECUTIVE_DOCKER_CLI_FAILS="${MAX_CONSECUTIVE_DOCKER_CLI_FAILS:-2}"
DOCKER_CLI_TIMEOUT="${DOCKER_CLI_TIMEOUT:-5}"
LOG_FILE="${LOG_FILE:-/tmp/docker_watchdog.log}"
LOG_MAX_BYTES="${LOG_MAX_BYTES:-209715200}"            # 200 MiB
DOCKER_SOCK="${DOCKER_SOCK:-/var/run/docker.sock}"
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-${DOCKER_ROOT:-/data}}"
PROXY_URL="${PROXY_URL:-http://httpproxy-headless.kubebrain.svc.pjlab.local:3128}"
NO_PROXY_LIST="${NO_PROXY_LIST:-localhost,127.0.0.1,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,.pjlab.local,.svc}"
PROXY_ENV_FILE="${PROXY_ENV_FILE:-/etc/seta_build_proxy.env}"
FIX_SCRIPT="${FIX_SCRIPT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/fix_dockerd_and_proxy.sh}"
WATCHDOG_AUTO_REPAIR="${WATCHDOG_AUTO_REPAIR:-1}"
WATCHDOG_REPAIR_MODE="${WATCHDOG_REPAIR_MODE:-restart}"  # restart | full-fix
WATCHDOG_FULL_FIX_ALLOW_SELF_STOP="${WATCHDOG_FULL_FIX_ALLOW_SELF_STOP:-0}"
WATCHDOG_KILL_SHIMS_ON_DOCKER_DOWN="${WATCHDOG_KILL_SHIMS_ON_DOCKER_DOWN:-1}"
REPAIR_LOCK_DIR="${REPAIR_LOCK_DIR:-/run/docker_watchdog_repair.lock}"
REPAIR_COOLDOWN_S="${REPAIR_COOLDOWN_S:-300}"

POOL_HOST="${POOL_HOST:-127.0.0.1}"
POOL_PORT="${POOL_PORT:-18081}"
POOL_PENDING_CLOSES_WARN="${POOL_PENDING_CLOSES_WARN:-50}"
POOL_PENDING_CLOSES_REPAIR="${POOL_PENDING_CLOSES_REPAIR:-1}"
POOL_PENDING_CLOSES_REPAIR_THRESHOLD="${POOL_PENDING_CLOSES_REPAIR_THRESHOLD:-${POOL_PENDING_CLOSES_WARN}}"
POOL_PENDING_CLOSES_STUCK_CHECKS="${POOL_PENDING_CLOSES_STUCK_CHECKS:-5}"
POOL_PENDING_CLOSES_ACTIVE_MAX="${POOL_PENDING_CLOSES_ACTIVE_MAX:-8}"
POOL_PENDING_CLOSES_REAP_LIMIT="${POOL_PENDING_CLOSES_REAP_LIMIT:-0}"
POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S="${POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S:-300}"
POOL_PENDING_CLOSES_CANCEL_API="${POOL_PENDING_CLOSES_CANCEL_API:-1}"
POOL_PENDING_CLOSES_CANCEL_TIMEOUT="${POOL_PENDING_CLOSES_CANCEL_TIMEOUT:-5}"
POOL_PENDING_CLOSES_CANCEL_MIN_AGE="${POOL_PENDING_CLOSES_CANCEL_MIN_AGE:-90}"
BRIDGE_NETS_WARN="${BRIDGE_NETS_WARN:-200}"
EMERGENCY_COOLDOWN_S="${EMERGENCY_COOLDOWN_S:-60}"
POOL_SERVER_NAME_REGEX="${POOL_SERVER_NAME_REGEX:-openclaw_pool_server}"
TASK_CONTAINER_REGEX="${TASK_CONTAINER_REGEX:-^[0-9]+[-_].*(slime[-_]?run|client|helper).*$}"
TASK_IMAGE_REGEX="${TASK_IMAGE_REGEX:-^tb__[0-9]+__.*(:|$)}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-600}"  # "I'm alive" line every 10 min

DISK_CHECK_INTERVAL="${DISK_CHECK_INTERVAL:-60}"
DISK_WARN_PCT="${DISK_WARN_PCT:-80}"
DISK_EMERGENCY_PCT="${DISK_EMERGENCY_PCT:-92}"
DISK_MIN_FREE_GB="${DISK_MIN_FREE_GB:-20}"
DISK_INODE_WARN_PCT="${DISK_INODE_WARN_PCT:-80}"
DISK_INODE_EMERGENCY_PCT="${DISK_INODE_EMERGENCY_PCT:-90}"
DISK_PRUNE_COOLDOWN_S="${DISK_PRUNE_COOLDOWN_S:-300}"
DISK_BUILD_CACHE_UNTIL="${DISK_BUILD_CACHE_UNTIL:-12h}"
WATCHDOG_AGGRESSIVE_IMAGE_PRUNE="${WATCHDOG_AGGRESSIVE_IMAGE_PRUNE:-0}"
WATCHDOG_PRUNE_TIMEOUT="${WATCHDOG_PRUNE_TIMEOUT:-120}"
POOL_STOP_ON_DISK_EMERGENCY="${POOL_STOP_ON_DISK_EMERGENCY:-1}"
POOL_STOP_COOLDOWN_S="${POOL_STOP_COOLDOWN_S:-300}"

LOG_PREFIX="[docker-watchdog]"

# ── 自身防 OOM ────────────────────────────────────────────────────────
echo -900 > /proc/self/oom_score_adj 2>/dev/null || true

# ── 状态 ──────────────────────────────────────────────────────────────
LAST_EMERGENCY_TS=0
LAST_DEEP_PROBE_TS=0
LAST_DISK_PRUNE_TS=0
LAST_POOL_STOP_TS=0
LAST_REPAIR_TS=0
LAST_PROC_WARN_TS=0
LAST_PIDS_RELIEF_TS=0
LAST_POOL_PENDING_REPAIR_TS=0
POOL_PENDING_HIGH_COUNT=0

# ── namespace 检测 ────────────────────────────────────────────────────
HOST_PID_NS=0
detect_pid_namespace() {
    # 与 PID 1 共享 mnt namespace 的进程基本就是 host
    local self_pid_ns host_pid_ns
    self_pid_ns=$(readlink /proc/self/ns/pid 2>/dev/null || echo "?")
    host_pid_ns=$(readlink /proc/1/ns/pid 2>/dev/null || echo "?")
    if [ -n "$self_pid_ns" ] && [ "$self_pid_ns" = "$host_pid_ns" ] && [ "$self_pid_ns" != "?" ]; then
        HOST_PID_NS=1
    else
        HOST_PID_NS=0
    fi
}

# ── 工具函数 ──────────────────────────────────────────────────────────
# 用 truncate-in-place 而不是 mv，否则 nohup 重定向的 fd 会丢
rotate_log_if_big() {
    [ -f "${LOG_FILE}" ] || return 0
    local sz
    sz=$(stat -c%s "${LOG_FILE}" 2>/dev/null || echo 0)
    [ -n "${sz}" ] && [ "${sz}" -ge 0 ] 2>/dev/null || sz=0
    [ "${sz}" -gt "${LOG_MAX_BYTES}" ] || return 0
    local tail_bytes=52428800   # 保留尾部 50 MB
    local tmp
    tmp=$(tail -c "$tail_bytes" "${LOG_FILE}" 2>/dev/null)
    : > "${LOG_FILE}"
    printf '%s\n' "$tmp" >> "${LOG_FILE}" 2>/dev/null || true
}

log() {
    echo "$(date '+%F %T') ${LOG_PREFIX} $*"
    rotate_log_if_big
}

positive_int_or_default() {
    local name="$1"
    local value="$2"
    local default="$3"
    if [[ "${value}" =~ ^[0-9]+$ ]] && [ "${value}" -gt 0 ] 2>/dev/null; then
        printf '%s' "${value}"
    else
        echo "$(date '+%F %T') ${LOG_PREFIX} WARN: invalid ${name}=${value}; using ${default}" >&2
        printf '%s' "${default}"
    fi
}

nonnegative_int_or_default() {
    local name="$1"
    local value="$2"
    local default="$3"
    if [[ "${value}" =~ ^[0-9]+$ ]]; then
        printf '%s' "${value}"
    else
        echo "$(date '+%F %T') ${LOG_PREFIX} WARN: invalid ${name}=${value}; using ${default}" >&2
        printf '%s' "${default}"
    fi
}

POOL_PENDING_CLOSES_WARN="$(positive_int_or_default POOL_PENDING_CLOSES_WARN "${POOL_PENDING_CLOSES_WARN}" 50)"
POOL_PENDING_CLOSES_REPAIR_THRESHOLD="$(positive_int_or_default POOL_PENDING_CLOSES_REPAIR_THRESHOLD "${POOL_PENDING_CLOSES_REPAIR_THRESHOLD}" "${POOL_PENDING_CLOSES_WARN}")"
POOL_PENDING_CLOSES_STUCK_CHECKS="$(positive_int_or_default POOL_PENDING_CLOSES_STUCK_CHECKS "${POOL_PENDING_CLOSES_STUCK_CHECKS}" 5)"
POOL_PENDING_CLOSES_ACTIVE_MAX="$(nonnegative_int_or_default POOL_PENDING_CLOSES_ACTIVE_MAX "${POOL_PENDING_CLOSES_ACTIVE_MAX}" 64)"
POOL_PENDING_CLOSES_REAP_LIMIT="$(nonnegative_int_or_default POOL_PENDING_CLOSES_REAP_LIMIT "${POOL_PENDING_CLOSES_REAP_LIMIT}" 0)"
POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S="$(nonnegative_int_or_default POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S "${POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S}" 300)"
POOL_PENDING_CLOSES_CANCEL_TIMEOUT="$(positive_int_or_default POOL_PENDING_CLOSES_CANCEL_TIMEOUT "${POOL_PENDING_CLOSES_CANCEL_TIMEOUT}" 5)"
POOL_PENDING_CLOSES_CANCEL_MIN_AGE="$(nonnegative_int_or_default POOL_PENDING_CLOSES_CANCEL_MIN_AGE "${POOL_PENDING_CLOSES_CANCEL_MIN_AGE}" 90)"

docker_alive() {
    timeout 3 curl -fsS --max-time 2 \
        --unix-socket "${DOCKER_SOCK}" \
        http://./_ping >/dev/null 2>&1
}

docker_cli_alive() {
    timeout "${DOCKER_CLI_TIMEOUT}" docker ps -q >/dev/null 2>&1
}

proxy_alive() {
    timeout 5 curl -fsS --max-time 4 --noproxy "" -x "${PROXY_URL}" http://example.com >/dev/null 2>&1
}

# 深度探活：模拟 pool_server 真实 reset 路径——能创建+删 bridge 网络
docker_deep_alive() {
    local netname="wd_probe_$(date +%s)_$$"
    if ! timeout 10 docker network create --driver bridge "$netname" >/dev/null 2>&1; then
        return 1
    fi
    timeout 5 docker network rm "$netname" >/dev/null 2>&1 || true
    return 0
}

# ── repair 防抖和互斥 ────────────────────────────────────────────────
acquire_repair_lock() {
    local now owner_pid owner_ts age
    now=$(date +%s)
    if mkdir "${REPAIR_LOCK_DIR}" 2>/dev/null; then
        printf '%s %s\n' "$$" "$now" > "${REPAIR_LOCK_DIR}/owner" 2>/dev/null || true
        return 0
    fi

    if [ -f "${REPAIR_LOCK_DIR}/owner" ]; then
        read -r owner_pid owner_ts < "${REPAIR_LOCK_DIR}/owner" 2>/dev/null || true
        age=$((now - ${owner_ts:-0}))
        if [ -n "${owner_pid:-}" ] && ! kill -0 "${owner_pid}" 2>/dev/null && [ "$age" -gt 600 ]; then
            log "REPAIR: removing stale lock ${REPAIR_LOCK_DIR} (owner=${owner_pid}, age=${age}s)"
            rm -rf "${REPAIR_LOCK_DIR}" 2>/dev/null || true
            if mkdir "${REPAIR_LOCK_DIR}" 2>/dev/null; then
                printf '%s %s\n' "$$" "$now" > "${REPAIR_LOCK_DIR}/owner" 2>/dev/null || true
                return 0
            fi
        fi
        log "REPAIR suppressed: another repair owns ${REPAIR_LOCK_DIR} (owner=${owner_pid:-?}, age=${age:-?}s)"
    else
        log "REPAIR suppressed: another repair owns ${REPAIR_LOCK_DIR}"
    fi
    return 1
}

release_repair_lock() {
    rm -rf "${REPAIR_LOCK_DIR}" 2>/dev/null || true
}

repair_snapshot() {
    log "REPAIR snapshot: pids=${LAST_PIDS_CUR:-?}/${LAST_PIDS_MAX:-?} (${LAST_PIDS_PCT:-?}%) tasks=${LAST_PROC_TASKS:-?} procs=${LAST_PROC_TOTAL:-?} zombies=${LAST_ZOMBIES:-?} dockerd=${LAST_DOCKERD_PROCS:-?} containerd=${LAST_CONTAINERD_PROCS:-?} shim=${LAST_SHIM_PROCS:-?} runc=${LAST_RUNC_PROCS:-?} docker_cli_fails=${DOCKER_CLI_FAILS:-0}"
}

run_full_fix_script() {
    if [ "${WATCHDOG_FULL_FIX_ALLOW_SELF_STOP}" != "1" ]; then
        log "REPAIR: full-fix requested but disabled because fix_dockerd_and_proxy.sh stops docker-watchdog; falling back to internal restart"
        restart_docker
        return $?
    fi
    if [ ! -f "${FIX_SCRIPT}" ]; then
        log "REPAIR: full-fix script not found: ${FIX_SCRIPT}; falling back to internal restart"
        restart_docker
        return $?
    fi
    log "REPAIR: running full fix script with START_WATCHDOG=0 SKIP_VERIFY=1 (this may stop this watchdog service)"
    DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" PROXY_URL="${PROXY_URL}" START_WATCHDOG=0 SKIP_VERIFY=1 \
        bash "${FIX_SCRIPT}"
}

trigger_repair() {
    local reason="$1"
    local force="${2:-0}"
    local now
    now=$(date +%s)
    if [ "${WATCHDOG_AUTO_REPAIR}" != "1" ]; then
        log "REPAIR disabled (WATCHDOG_AUTO_REPAIR=0): ${reason}"
        repair_snapshot
        return 0
    fi
    if [ "${force}" != "1" ] && [ $((now - LAST_REPAIR_TS)) -lt "${REPAIR_COOLDOWN_S}" ]; then
        log "REPAIR suppressed (cooldown ${REPAIR_COOLDOWN_S}s active): ${reason}"
        repair_snapshot
        return 0
    fi
    if ! acquire_repair_lock; then
        return 0
    fi

    LAST_REPAIR_TS="$now"
    log "REPAIR trigger: ${reason}$([ "${force}" = "1" ] && echo " (forced)" || true)"
    repair_snapshot
    case "${WATCHDOG_REPAIR_MODE}" in
        full-fix)
            run_full_fix_script || log "REPAIR: full-fix/restart path failed"
            ;;
        restart|*)
            restart_docker || log "REPAIR: internal dockerd restart failed"
            ;;
    esac
    release_repair_lock
}

task_container_lines() {
    timeout "${DOCKER_CLI_TIMEOUT}" docker ps --format '{{.ID}}\t{{.Names}}\t{{.Image}}' 2>/dev/null \
        | awk -F '\t' -v name_re="${TASK_CONTAINER_REGEX}" -v image_re="${TASK_IMAGE_REGEX}" '
            $2 ~ name_re || $3 ~ image_re {print $0}
        '
}

task_container_ids() {
    task_container_lines | awk -F '\t' '{print $1}'
}

task_container_count() {
    task_container_lines | wc -l
}

stop_pool_server_for_pressure() {
    local reason="$1"
    local proc_dir pid cmdline killed=0

    for proc_dir in /proc/[0-9]*; do
        [ -r "${proc_dir}/cmdline" ] || continue
        pid="${proc_dir##*/}"
        cmdline="$(< "${proc_dir}/cmdline")"
        case "${cmdline}" in
            *terminal-rl.remote.pool_server*|*remote.pool_server*|*pool_server.py*|*run_pool_server_pu_v2.sh*)
                log "PRESSURE: stopping pool_server pid=${pid} reason=${reason}"
                kill "${pid}" 2>/dev/null || true
                killed=$((killed + 1))
                ;;
        esac
    done

    if [ "${killed}" -eq 0 ]; then
        log "PRESSURE: no pool_server process matched for stop (reason=${reason})"
    fi
}

kill_task_containers_for_pressure() {
    local reason="$1"
    local limit="${2:-0}"
    local ids n

    if [ "${limit}" -gt 0 ] 2>/dev/null; then
        ids="$(task_container_ids | head -n "${limit}" 2>/dev/null || true)"
    else
        ids="$(task_container_ids 2>/dev/null || true)"
    fi
    if [ -z "${ids}" ]; then
        log "PRESSURE: no task containers matched for kill (reason=${reason}, name_re=${TASK_CONTAINER_REGEX}, image_re=${TASK_IMAGE_REGEX})"
        return 1
    fi

    n="$(printf '%s\n' "${ids}" | wc -l)"
    log "PRESSURE: removing ${n} task containers (reason=${reason})"
    printf '%s\n' "${ids}" | xargs -r -n 10 timeout 30 docker rm -f >/dev/null 2>&1 || true
    return 0
}

pids_pressure_relief() {
    local reason="$1"
    local now
    now=$(date +%s)
    if [ $((now - LAST_PIDS_RELIEF_TS)) -lt "${PIDS_RELIEF_COOLDOWN_S}" ]; then
        log "PRESSURE suppressed (cooldown ${PIDS_RELIEF_COOLDOWN_S}s active): ${reason}"
        return 0
    fi
    LAST_PIDS_RELIEF_TS="$now"

    log "PRESSURE: pids emergency relief: ${reason}"
    repair_snapshot
    stop_pool_server_for_pressure "${reason}"
    kill_task_containers_for_pressure "${reason}" 0 || true
}

repair_stuck_pool_pending_closes() {
    local pending="$1"
    local active="$2"
    local now reason matched repair_tmp repair_code

    [ "${POOL_PENDING_CLOSES_REPAIR}" = "1" ] || return 0
    now=$(date +%s)
    if [ $((now - LAST_POOL_PENDING_REPAIR_TS)) -lt "${POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S}" ]; then
        log "POOL_REPAIR suppressed (cooldown ${POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S}s active): pending_closes=${pending} active=${active}"
        return 0
    fi
    LAST_POOL_PENDING_REPAIR_TS="$now"

    reason="stuck pool pending_closes=${pending} active=${active} high_count=${POOL_PENDING_HIGH_COUNT}"
    log "POOL_REPAIR: ${reason}; reaping task containers with broad matcher"
    matched=1
    kill_task_containers_for_pressure "${reason}" "${POOL_PENDING_CLOSES_REAP_LIMIT}" || matched=0
    timeout 30 docker container prune -f --filter "until=0s" >/dev/null 2>&1 || true
    timeout 30 docker network prune -f >/dev/null 2>&1 || true

    if [ "${POOL_PENDING_CLOSES_CANCEL_API}" = "1" ]; then
        repair_tmp="$(mktemp /tmp/pool_pending_repair.XXXXXX 2>/dev/null || echo /tmp/pool_pending_repair.$$)"
        repair_code=$(timeout 10 curl -sS --noproxy '*' -o "${repair_tmp}" -w '%{http_code}' \
            -X POST -H 'Content-Type: application/json' \
            --data "{\"reason\":\"watchdog_pending_closes_repair\",\"max_active_runs\":${POOL_PENDING_CLOSES_ACTIVE_MAX},\"cancel_timeout\":${POOL_PENDING_CLOSES_CANCEL_TIMEOUT},\"min_age\":${POOL_PENDING_CLOSES_CANCEL_MIN_AGE}}" \
            "http://${POOL_HOST}:${POOL_PORT}/repair/pending_closes" 2>/dev/null || echo "000")
        if [ "${repair_code}" = "200" ]; then
            log "POOL_REPAIR: pending-close API response: $(head -c 300 "${repair_tmp}" 2>/dev/null)"
        else
            log "POOL_REPAIR: pending-close API failed HTTP ${repair_code}: $(head -c 300 "${repair_tmp}" 2>/dev/null)"
            if [ "${matched}" = "0" ] && [ "${active}" -eq 0 ] 2>/dev/null; then
                log "POOL_REPAIR: no task containers matched and active=0; stop/restart pool_server manually if pending_closes remains high"
            fi
        fi
        rm -f "${repair_tmp}" 2>/dev/null || true
    fi
}

# ── 紧急泄压（带冷却 + foreground + timeout）─────────────────────────
emergency_pressure_relief() {
    local reason="$1"
    local now
    now=$(date +%s)
    if [ $((now - LAST_EMERGENCY_TS)) -lt "${EMERGENCY_COOLDOWN_S}" ]; then
        log "EMERGENCY suppressed (cooldown ${EMERGENCY_COOLDOWN_S}s active): ${reason}"
        return
    fi
    LAST_EMERGENCY_TS="$now"
    log "EMERGENCY: ${reason} — kill task containers + prune"

    kill_task_containers_for_pressure "${reason}" 30 || true

    # 清理 stopped + dangling network（foreground，防并发拖死 dockerd）
    timeout 30 docker container prune -f >/dev/null 2>&1 || true
    timeout 30 docker network prune -f >/dev/null 2>&1 || true
}

# ── cgroup 检测（v1 + v2）────────────────────────────────────────────
CGROUP_VERSION=""
CGROUP_PIDS_DIR=""
CGROUP_MEM_DIR=""
CGROUP_PIDS_MAX_VAL=""
CGROUP_MEM_MAX_VAL=""
CGROUP_PIDS_CUR_FILE=""
CGROUP_MEM_CUR_FILE=""

# 沿 cgroup 路径从深到浅扫描，找到"最严有限限制"所在的目录
# 参数: $1=控制器挂载点  $2=cgroup 相对路径  $3=current 文件名  $4=max 文件名
find_tightest_limit() {
    local mount="$1" rel="$2" cur_name="$3" max_name="$4"
    local best_dir="" best_val=""
    local path="$rel"
    while [ -n "$path" ] && [ "$path" != "/" ]; do
        local full="${mount}${path}"
        if [ -f "${full}/${max_name}" ]; then
            local v
            v=$(cat "${full}/${max_name}" 2>/dev/null)
            if [ -n "$v" ] && [ "$v" != "max" ] \
               && [ "$v" -gt 0 ] 2>/dev/null \
               && [ "$v" -lt 9000000000000000000 ] 2>/dev/null; then
                if [ -z "$best_val" ] || [ "$v" -lt "$best_val" ] 2>/dev/null; then
                    best_val="$v"
                    best_dir="$full"
                fi
            fi
        fi
        path="${path%/*}"
    done
    [ -n "$best_dir" ] && echo "${best_dir}|${best_val}"
}

detect_cgroup_v2() {
    [ -f /sys/fs/cgroup/cgroup.controllers ] || return 1
    local rel
    rel=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)
    [ -n "$rel" ] || return 1
    # v2: pids.max / memory.max 在同一 unified hierarchy
    local r
    r=$(find_tightest_limit /sys/fs/cgroup "$rel" pids.current pids.max)
    if [ -n "$r" ]; then
        CGROUP_PIDS_DIR="${r%|*}"
        CGROUP_PIDS_MAX_VAL="${r#*|}"
        CGROUP_PIDS_CUR_FILE="${CGROUP_PIDS_DIR}/pids.current"
    fi
    r=$(find_tightest_limit /sys/fs/cgroup "$rel" memory.current memory.max)
    if [ -n "$r" ]; then
        CGROUP_MEM_DIR="${r%|*}"
        CGROUP_MEM_MAX_VAL="${r#*|}"
        CGROUP_MEM_CUR_FILE="${CGROUP_MEM_DIR}/memory.current"
    fi
    if [ -n "$CGROUP_PIDS_DIR" ] || [ -n "$CGROUP_MEM_DIR" ]; then
        CGROUP_VERSION="v2"
        return 0
    fi
    return 1
}

detect_cgroup_v1() {
    local pids_rel mem_rel
    pids_rel=$(awk -F: '$2 ~ /(^|,)pids(,|$)/  {print $3}'   /proc/self/cgroup 2>/dev/null | head -1)
    mem_rel=$(awk  -F: '$2 ~ /(^|,)memory(,|$)/{print $3}'   /proc/self/cgroup 2>/dev/null | head -1)
    if [ -d /sys/fs/cgroup/pids ] && [ -n "$pids_rel" ]; then
        local r
        r=$(find_tightest_limit /sys/fs/cgroup/pids "$pids_rel" pids.current pids.max)
        if [ -n "$r" ]; then
            CGROUP_PIDS_DIR="${r%|*}"
            CGROUP_PIDS_MAX_VAL="${r#*|}"
            CGROUP_PIDS_CUR_FILE="${CGROUP_PIDS_DIR}/pids.current"
        fi
    fi
    if [ -d /sys/fs/cgroup/memory ] && [ -n "$mem_rel" ]; then
        local r
        r=$(find_tightest_limit /sys/fs/cgroup/memory "$mem_rel" memory.usage_in_bytes memory.limit_in_bytes)
        if [ -n "$r" ]; then
            CGROUP_MEM_DIR="${r%|*}"
            CGROUP_MEM_MAX_VAL="${r#*|}"
            CGROUP_MEM_CUR_FILE="${CGROUP_MEM_DIR}/memory.usage_in_bytes"
        fi
    fi
    if [ -n "$CGROUP_PIDS_DIR" ] || [ -n "$CGROUP_MEM_DIR" ]; then
        CGROUP_VERSION="v1"
        return 0
    fi
    return 1
}

detect_cgroup() {
    detect_cgroup_v2 || detect_cgroup_v1 || return 1
}

read_effective_pids() {
    local cur="" max=""
    if [ -n "${CGROUP_PIDS_CUR_FILE}" ] && [ -f "${CGROUP_PIDS_CUR_FILE}" ]; then
        read -r cur < "${CGROUP_PIDS_CUR_FILE}" 2>/dev/null || cur=""
        max="${CGROUP_PIDS_MAX_VAL}"
    fi
    if [ -z "$cur" ]; then
        cur="${LAST_PROC_TASKS:-}"
    fi
    if [ -z "$max" ] && [ -f /proc/sys/kernel/threads-max ]; then
        read -r max < /proc/sys/kernel/threads-max 2>/dev/null || max=""
    fi
    [ -n "$cur" ] && [ -n "$max" ] && echo "${cur} ${max}"
}

collect_proc_metrics() {
    local proc_dir task_dir stat_line rest state name
    local total=0 tasks=0 zombies=0 dockerd=0 containerd=0 shim=0 runc=0 docker_cli=0

    for proc_dir in /proc/[0-9]*; do
        [ -d "$proc_dir" ] || continue
        total=$((total + 1))
        if IFS= read -r name < "${proc_dir}/comm" 2>/dev/null; then
            :
        else
            name="?"
        fi
        if IFS= read -r stat_line < "${proc_dir}/stat" 2>/dev/null; then
            rest="${stat_line#*) }"
            state="${rest%% *}"
            [ "$state" = "Z" ] && zombies=$((zombies + 1))
        fi
        for task_dir in "${proc_dir}"/task/[0-9]*; do
            [ -d "$task_dir" ] && tasks=$((tasks + 1))
        done
        case "$name" in
            dockerd) dockerd=$((dockerd + 1)) ;;
            containerd) containerd=$((containerd + 1)) ;;
            containerd-shim*) shim=$((shim + 1)) ;;
            runc) runc=$((runc + 1)) ;;
            docker) docker_cli=$((docker_cli + 1)) ;;
        esac
    done

    LAST_PROC_TOTAL="$total"
    LAST_PROC_TASKS="$tasks"
    LAST_ZOMBIES="$zombies"
    LAST_DOCKERD_PROCS="$dockerd"
    LAST_CONTAINERD_PROCS="$containerd"
    LAST_SHIM_PROCS="$shim"
    LAST_RUNC_PROCS="$runc"
    LAST_DOCKER_CLI_PROCS="$docker_cli"

    local pids cur max pct
    pids="$(read_effective_pids 2>/dev/null || true)"
    cur="${pids%% *}"
    max="${pids##* }"
    pct="?"
    if [ -n "$cur" ] && [ -n "$max" ] && [ "$max" -gt 0 ] 2>/dev/null; then
        pct=$((cur * 100 / max))
    fi
    LAST_PIDS_CUR="$cur"
    LAST_PIDS_MAX="$max"
    LAST_PIDS_PCT="$pct"
}

proc_warn_log() {
    local now msg
    msg="$1"
    now=$(date +%s)
    if [ $((now - LAST_PROC_WARN_TS)) -ge "${PROC_WARN_COOLDOWN_S}" ]; then
        log "$msg"
        LAST_PROC_WARN_TS="$now"
    fi
}

monitor_proc_pressure() {
    collect_proc_metrics

    local docker_related
    docker_related=$((LAST_DOCKERD_PROCS + LAST_CONTAINERD_PROCS + LAST_SHIM_PROCS + LAST_RUNC_PROCS + LAST_DOCKER_CLI_PROCS))

    if [ "${LAST_PIDS_PCT}" != "?" ] && [ "${LAST_PIDS_PCT}" -ge "${PIDS_EMERGENCY_PCT}" ] 2>/dev/null; then
        pids_pressure_relief "pids pressure ${LAST_PIDS_CUR}/${LAST_PIDS_MAX} (${LAST_PIDS_PCT}%) before fork failure"
        trigger_repair "pids pressure ${LAST_PIDS_CUR}/${LAST_PIDS_MAX} (${LAST_PIDS_PCT}%) before fork failure"
        return 0
    fi
    if [ "${LAST_ZOMBIES}" -ge "${ZOMBIE_EMERGENCY}" ] 2>/dev/null; then
        trigger_repair "zombie process pressure zombies=${LAST_ZOMBIES}"
        return 0
    fi
    if [ "${LAST_SHIM_PROCS}" -ge "${SHIM_PROC_EMERGENCY}" ] 2>/dev/null; then
        trigger_repair "containerd-shim process pressure shim=${LAST_SHIM_PROCS}"
        return 0
    fi
    if [ "${LAST_RUNC_PROCS}" -ge "${RUNC_PROC_EMERGENCY}" ] 2>/dev/null; then
        trigger_repair "runc process pressure runc=${LAST_RUNC_PROCS}"
        return 0
    fi
    if [ "${docker_related}" -ge "${DOCKER_PROC_EMERGENCY}" ] 2>/dev/null; then
        trigger_repair "Docker-related process pressure docker_related=${docker_related}"
        return 0
    fi

    if [ "${LAST_PIDS_PCT}" != "?" ] && [ "${LAST_PIDS_PCT}" -ge "${PIDS_WARN_PCT}" ] 2>/dev/null; then
        proc_warn_log "WARN: pids ${LAST_PIDS_CUR}/${LAST_PIDS_MAX} (${LAST_PIDS_PCT}%) tasks=${LAST_PROC_TASKS} procs=${LAST_PROC_TOTAL} zombies=${LAST_ZOMBIES} shim=${LAST_SHIM_PROCS} runc=${LAST_RUNC_PROCS}"
    elif [ "${LAST_ZOMBIES}" -ge "${ZOMBIE_WARN}" ] 2>/dev/null \
       || [ "${LAST_SHIM_PROCS}" -ge "${SHIM_PROC_WARN}" ] 2>/dev/null \
       || [ "${LAST_RUNC_PROCS}" -ge "${RUNC_PROC_WARN}" ] 2>/dev/null \
       || [ "${docker_related}" -ge "${DOCKER_PROC_WARN}" ] 2>/dev/null; then
        proc_warn_log "WARN: process pressure tasks=${LAST_PROC_TASKS} procs=${LAST_PROC_TOTAL} zombies=${LAST_ZOMBIES} docker_related=${docker_related} dockerd=${LAST_DOCKERD_PROCS} containerd=${LAST_CONTAINERD_PROCS} shim=${LAST_SHIM_PROCS} runc=${LAST_RUNC_PROCS} docker_cli=${LAST_DOCKER_CLI_PROCS}"
    fi
}

monitor_pod_cgroup() {
    [ -z "$CGROUP_VERSION" ] && return 0

    if [ -n "$CGROUP_PIDS_DIR" ] && [ -f "$CGROUP_PIDS_CUR_FILE" ]; then
        local cur
        read -r cur < "$CGROUP_PIDS_CUR_FILE" 2>/dev/null
        if [ -n "$cur" ] && [ "$cur" -ge 0 ] 2>/dev/null; then
            local pct=$(( cur * 100 / CGROUP_PIDS_MAX_VAL ))
            if [ "$pct" -ge "$PIDS_EMERGENCY_PCT" ]; then
                LAST_PIDS_CUR="$cur"
                LAST_PIDS_MAX="$CGROUP_PIDS_MAX_VAL"
                LAST_PIDS_PCT="$pct"
                pids_pressure_relief "cgroup PIDs ${cur}/${CGROUP_PIDS_MAX_VAL} (${pct}%)"
                trigger_repair "cgroup PIDs ${cur}/${CGROUP_PIDS_MAX_VAL} (${pct}%)"
            elif [ "$pct" -ge "$PIDS_WARN_PCT" ]; then
                log "WARN: PIDs ${cur}/${CGROUP_PIDS_MAX_VAL} (${pct}%) — aggressive cleanup"
                timeout 20 docker container prune -f --filter "until=30s" >/dev/null 2>&1 || true
            fi
        fi
    fi

    if [ -n "$CGROUP_MEM_DIR" ] && [ -f "$CGROUP_MEM_CUR_FILE" ]; then
        local cur
        read -r cur < "$CGROUP_MEM_CUR_FILE" 2>/dev/null
        if [ -n "$cur" ] && [ "$cur" -ge 0 ] 2>/dev/null; then
            local pct=$(( cur * 100 / CGROUP_MEM_MAX_VAL ))
            if [ "$pct" -ge "$MEM_EMERGENCY_PCT" ]; then
                emergency_pressure_relief "Memory ${cur}/${CGROUP_MEM_MAX_VAL} (${pct}%)"
            elif [ "$pct" -ge "$MEM_WARN_PCT" ]; then
                log "WARN: Memory ${cur}/${CGROUP_MEM_MAX_VAL} (${pct}%) — pruning"
                timeout 20 docker container prune -f --filter "until=30s" >/dev/null 2>&1 || true
            fi
        fi
    fi
}

# ── pool_server 监控（核心修复：捕捉 dockerd OK 但 /reset 500 的故障形态）──
# 副作用：每次成功调用会更新 LAST_POOL_ACTIVE / LAST_POOL_PENDING / LAST_BRIDGE_NETS
# 供 heartbeat 复用，不重新发起 HTTP/docker 调用。
LAST_POOL_ACTIVE="?"
LAST_POOL_PENDING="?"
LAST_BRIDGE_NETS="?"
check_pool_server() {
    local health_tmp health_code
    health_tmp="$(mktemp /tmp/pool_health.XXXXXX 2>/dev/null || echo /tmp/pool_health.$$)"
    health_code=$(timeout 5 curl -sS --noproxy '*' -o "$health_tmp" -w '%{http_code}' \
        "http://${POOL_HOST}:${POOL_PORT}/healthz" 2>/dev/null || echo "000")
    if [ "$health_code" = "000" ]; then
        log "WARN: pool_server /healthz unreachable"
        LAST_POOL_ACTIVE="down"
        LAST_POOL_PENDING="down"
        rm -f "$health_tmp" 2>/dev/null || true
        return 1
    fi
    if [ "$health_code" -ge 400 ] 2>/dev/null; then
        log "WARN: pool_server /healthz returned HTTP ${health_code}: $(head -c 300 "$health_tmp" 2>/dev/null)"
    fi
    rm -f "$health_tmp" 2>/dev/null || true

    local body pending=0 active=0 active_tasks=0 active_runs=0
    body=$(timeout 3 curl -fsS --noproxy '*' "http://${POOL_HOST}:${POOL_PORT}/status" 2>/dev/null)
    if [ -n "$body" ]; then
        # 合并到 1 个 python 调用减少 fork 开销
        local parsed
        parsed=$(echo "$body" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    p = d.get("pool", d)
    print(p.get("pending_closes", 0), p.get("active_tasks", 0), p.get("total_active_runs", 0))
except Exception:
    print("0 0 0")
' 2>/dev/null)
        set -- ${parsed}
        pending="${1:-0}"
        active_tasks="${2:-0}"
        active_runs="${3:-0}"
        active="${active_runs}"
        pending="${pending:-0}"
        active="${active:-0}"
        LAST_POOL_PENDING="$pending"
        LAST_POOL_ACTIVE="$active"
        if [ "$pending" -gt "$POOL_PENDING_CLOSES_WARN" ] 2>/dev/null; then
            POOL_PENDING_HIGH_COUNT=$((POOL_PENDING_HIGH_COUNT + 1))
            log "WARN: pool_server pending_closes=${pending} (active_runs=${active_runs}, active_tasks=${active_tasks}, high_count=${POOL_PENDING_HIGH_COUNT}/${POOL_PENDING_CLOSES_STUCK_CHECKS})"
            if [ "$pending" -ge "$POOL_PENDING_CLOSES_REPAIR_THRESHOLD" ] 2>/dev/null \
               && [ "$POOL_PENDING_HIGH_COUNT" -ge "$POOL_PENDING_CLOSES_STUCK_CHECKS" ] 2>/dev/null \
               && [ "$active" -le "$POOL_PENDING_CLOSES_ACTIVE_MAX" ] 2>/dev/null; then
                repair_stuck_pool_pending_closes "$pending" "$active"
                POOL_PENDING_HIGH_COUNT=0
            fi
        else
            POOL_PENDING_HIGH_COUNT=0
        fi
    fi

    local nets
    nets=$(docker network ls --filter driver=bridge -q 2>/dev/null | wc -l)
    LAST_BRIDGE_NETS="$nets"
    if [ "$nets" -gt "$BRIDGE_NETS_WARN" ] 2>/dev/null; then
        log "WARN: ${nets} bridge networks, address-pool risk; pruning"
        timeout 30 docker network prune -f >/dev/null 2>&1 || true
    fi
    return 0
}

monitor_docker_cli() {
    if docker_cli_alive; then
        DOCKER_CLI_FAILS=0
        LAST_DOCKER_CLI_STATUS="ok"
        return 0
    fi

    DOCKER_CLI_FAILS=$((DOCKER_CLI_FAILS + 1))
    LAST_DOCKER_CLI_STATUS="fail"
    log "WARN: docker CLI probe timed out or failed (${DOCKER_CLI_FAILS}/${MAX_CONSECUTIVE_DOCKER_CLI_FAILS}, timeout=${DOCKER_CLI_TIMEOUT}s)"
    if [ "${DOCKER_CLI_FAILS}" -ge "${MAX_CONSECUTIVE_DOCKER_CLI_FAILS}" ]; then
        trigger_repair "docker CLI timeout/failure while dockerd ping may still be ambiguous"
        DOCKER_CLI_FAILS=0
    fi
}

monitor_proxy() {
    if proxy_alive; then
        LAST_PROXY_STATUS="ok"
    else
        LAST_PROXY_STATUS="fail"
        log "WARN: proxy probe failed: ${PROXY_URL}"
    fi
}

stop_pool_server_for_disk_pressure() {
    [ "${POOL_STOP_ON_DISK_EMERGENCY}" = "1" ] || return 0
    local now pids
    now=$(date +%s)
    if [ $((now - LAST_POOL_STOP_TS)) -lt "${POOL_STOP_COOLDOWN_S}" ]; then
        log "DISK: pool_server protective stop suppressed by cooldown"
        return 0
    fi
    LAST_POOL_STOP_TS="$now"

    pids=$(pgrep -f "terminal-rl.remote.pool_server|remote.pool_server" 2>/dev/null || true)
    if [ -z "$pids" ]; then
        log "DISK: pool_server already stopped or not found"
        return 0
    fi

    log "DISK: protective stop of pool_server due to persistent Docker data-root pressure: pid(s) ${pids}"
    echo "$pids" | xargs -r kill 2>/dev/null || true
    sleep 5
    echo "$pids" | xargs -r kill -9 2>/dev/null || true
}

# ── stopped 容器清理 ─────────────────────────────────────────────────
cleanup_stopped() {
    local stopped
    stopped=$(docker ps -aq --filter "status=exited" --filter "status=dead" 2>/dev/null | wc -l)
    if [ "${stopped}" -gt 5 ]; then
        log "Cleaning ${stopped} stopped containers..."
        timeout 30 docker container prune -f --filter "until=2m" >/dev/null 2>&1 || true
    fi
    local dn
    dn=$(docker network ls --filter "dangling=true" -q 2>/dev/null | wc -l)
    if [ "${dn}" -gt 0 ]; then
        timeout 20 docker network prune -f >/dev/null 2>&1 || true
    fi
}

# ── Docker data-root 磁盘压力监控 ─────────────────────────────────────
# 不调用 docker system df：在 image/cache 很多或 dockerd 元数据锁竞争时它会卡很久。
# 这里只用 df 快速判断 /data 是否接近爆盘，再做带 timeout 的渐进清理。
docker_disk_stats() {
    local line used avail inode_line inode_used
    line=$(df -P -BG "${DOCKER_DATA_ROOT}" 2>/dev/null | awk 'NR==2 {print $5, $4}')
    [ -n "$line" ] || return 1
    used="${line% *}"
    avail="${line#* }"
    used="${used%\%}"
    avail="${avail%G}"
    inode_line=$(df -Pi "${DOCKER_DATA_ROOT}" 2>/dev/null | awk 'NR==2 {print $5}')
    inode_used="${inode_line%\%}"
    [ -n "$used" ] && [ -n "$avail" ] && [ -n "$inode_used" ] || return 1
    echo "${used} ${avail} ${inode_used}"
}

disk_prune_light() {
    log "DISK: light cleanup: stopped containers + networks + build cache older than ${DISK_BUILD_CACHE_UNTIL}"
    timeout 30 docker container prune -f --filter "until=2m" >/dev/null 2>&1 || true
    timeout 30 docker network prune -f >/dev/null 2>&1 || true
    timeout "${WATCHDOG_PRUNE_TIMEOUT}" docker builder prune -af --filter "until=${DISK_BUILD_CACHE_UNTIL}" >/dev/null 2>&1 || true
    timeout 60 docker image prune -f >/dev/null 2>&1 || true
}

disk_prune_emergency() {
    local reason="$1"
    emergency_pressure_relief "Docker data-root disk pressure: ${reason}"
    disk_prune_light
    if [ "${WATCHDOG_AGGRESSIVE_IMAGE_PRUNE}" = "1" ]; then
        log "DISK: WATCHDOG_AGGRESSIVE_IMAGE_PRUNE=1, pruning all unused images"
        timeout "${WATCHDOG_PRUNE_TIMEOUT}" docker image prune -af >/dev/null 2>&1 || true
    else
        log "DISK: aggressive unused-image prune disabled; set WATCHDOG_AGGRESSIVE_IMAGE_PRUNE=1 to enable"
    fi

    local stats used_pct avail_gb inode_pct
    stats=$(docker_disk_stats 2>/dev/null || true)
    if [ -z "$stats" ]; then
        log "DISK: cannot read stats after emergency cleanup; stopping pool_server defensively"
        stop_pool_server_for_disk_pressure
        return 0
    fi
    used_pct=$(echo "$stats" | awk '{print $1}')
    avail_gb=$(echo "$stats" | awk '{print $2}')
    inode_pct=$(echo "$stats" | awk '{print $3}')
    if [ "${used_pct}" -ge "${DISK_EMERGENCY_PCT}" ] 2>/dev/null \
       || [ "${avail_gb}" -le "${DISK_MIN_FREE_GB}" ] 2>/dev/null \
       || [ "${inode_pct}" -ge "${DISK_INODE_EMERGENCY_PCT}" ] 2>/dev/null; then
        log "DISK: pressure persists after cleanup (${used_pct}% used, ${avail_gb}GB free, inode ${inode_pct}%); stopping pool_server"
        stop_pool_server_for_disk_pressure
    fi
}

monitor_docker_disk() {
    local stats used_pct avail_gb inode_pct now
    stats=$(docker_disk_stats) || {
        log "WARN: cannot read disk stats for ${DOCKER_DATA_ROOT}"
        return 0
    }
    used_pct=$(echo "$stats" | awk '{print $1}')
    avail_gb=$(echo "$stats" | awk '{print $2}')
    inode_pct=$(echo "$stats" | awk '{print $3}')

    if [ "${used_pct}" -ge "${DISK_EMERGENCY_PCT}" ] 2>/dev/null \
       || [ "${avail_gb}" -le "${DISK_MIN_FREE_GB}" ] 2>/dev/null \
       || [ "${inode_pct}" -ge "${DISK_INODE_EMERGENCY_PCT}" ] 2>/dev/null; then
        now=$(date +%s)
        if [ $((now - LAST_DISK_PRUNE_TS)) -lt "${DISK_PRUNE_COOLDOWN_S}" ]; then
            log "DISK: emergency condition persists (${used_pct}% used, ${avail_gb}GB free, inode ${inode_pct}%), cleanup cooldown active"
            stop_pool_server_for_disk_pressure
            return 0
        fi
        LAST_DISK_PRUNE_TS="$now"
        disk_prune_emergency "${used_pct}% used, ${avail_gb}GB free, inode ${inode_pct}%"
        return 0
    fi

    if [ "${used_pct}" -ge "${DISK_WARN_PCT}" ] 2>/dev/null \
       || [ "${inode_pct}" -ge "${DISK_INODE_WARN_PCT}" ] 2>/dev/null; then
        now=$(date +%s)
        if [ $((now - LAST_DISK_PRUNE_TS)) -lt "${DISK_PRUNE_COOLDOWN_S}" ]; then
            log "DISK: warn ${used_pct}% used, ${avail_gb}GB free, inode ${inode_pct}%; cleanup cooldown active"
            return 0
        fi
        LAST_DISK_PRUNE_TS="$now"
        log "DISK: warn ${used_pct}% used, ${avail_gb}GB free, inode ${inode_pct}%"
        disk_prune_light
    fi
}

# ── 运行容器数上限（双闸门，排除 pool_server）─────────────────────────
# 副作用：更新 LAST_RUNNING_TASKS 供 heartbeat 复用
LAST_RUNNING_TASKS="?"
enforce_container_limit() {
    local running
    # 只统计 task 容器（带数字前缀 + client/helper 后缀），不算 pool_server 等基础容器
    running=$(task_container_count 2>/dev/null || echo 0)
    LAST_RUNNING_TASKS="$running"

    if [ "${running}" -gt "${HARD_KILL_THRESHOLD}" ]; then
        local excess=$((running - MAX_RUNNING_CONTAINERS))
        log "HARD LIMIT: ${running} task containers > ${HARD_KILL_THRESHOLD}, killing ${excess} oldest"
        kill_task_containers_for_pressure "hard task container limit ${running}>${HARD_KILL_THRESHOLD}" "${excess}" || true
        return
    fi

    if [ "${running}" -gt "${MAX_RUNNING_CONTAINERS}" ]; then
        local excess=$((running - MAX_RUNNING_CONTAINERS))
        log "Soft limit: ${running} task containers > ${MAX_RUNNING_CONTAINERS}, killing ${excess} oldest"
        kill_task_containers_for_pressure "soft task container limit ${running}>${MAX_RUNNING_CONTAINERS}" "${excess}" || true
    fi
}

# ── dockerd 重启（绕过 systemctl restart，沿用 restart_docker_force.sh 模式）──
restart_docker() {
    log "Docker daemon is DOWN. Attempting forced restart (no systemctl restart)..."
    collect_proc_metrics
    repair_snapshot

    if [ "${LAST_SHIM_PROCS:-0}" -ge "${DOCKER_DOWN_SHIM_RELIEF}" ] 2>/dev/null; then
        log "Docker is down with ${LAST_SHIM_PROCS} containerd-shim processes (threshold=${DOCKER_DOWN_SHIM_RELIEF}); stopping pool_server before restart"
        stop_pool_server_for_pressure "dockerd down with shim pressure"
    fi

    # 1) 阻断 systemd auto-restart：reset-failed + stop docker.socket
    timeout 5 systemctl reset-failed docker.service docker.socket 2>/dev/null || true
    timeout 5 systemctl stop docker.socket 2>/dev/null || true

    # 2) pkill -9 dockerd; optionally clear shim processes once Docker is confirmed down.
    pkill -9 -x dockerd 2>/dev/null || true
    sleep 2
    if pgrep -x dockerd >/dev/null 2>&1; then
        log "WARN: dockerd still alive after SIGKILL (D state?), aborting state cleanup"
        return 1
    fi

    if [ "${WATCHDOG_KILL_SHIMS_ON_DOCKER_DOWN}" = "1" ] \
       && [ "${LAST_SHIM_PROCS:-0}" -ge "${DOCKER_DOWN_SHIM_RELIEF}" ] 2>/dev/null; then
        local shim_pids shim_n
        shim_pids="$(pgrep -f containerd-shim 2>/dev/null || true)"
        if [ -n "${shim_pids}" ]; then
            shim_n="$(printf '%s\n' "${shim_pids}" | wc -l)"
            log "Killing ${shim_n} containerd-shim processes because dockerd is down and shim pressure is high"
            printf '%s\n' "${shim_pids}" | xargs -r kill -9 2>/dev/null || true
            sleep 1
        fi
    fi

    rm -f /var/run/docker.pid "${DOCKER_SOCK}"

    # 3) 清理 container state。正常情况下仅在 no-shim 时清；Docker 已 down 且 shim 压力高时，
    #    这些 state 已无法安全恢复，清理后让 dockerd 干净启动。
    if [ "$HOST_PID_NS" = "1" ]; then
        if ! pgrep -f containerd-shim >/dev/null 2>&1 \
           || { [ "${WATCHDOG_KILL_SHIMS_ON_DOCKER_DOWN}" = "1" ] \
                && [ "${LAST_SHIM_PROCS:-0}" -ge "${DOCKER_DOWN_SHIM_RELIEF}" ] 2>/dev/null; }; then
            if [ -d "${DOCKER_DATA_ROOT}/containers" ]; then
                local n
                n=$(find "${DOCKER_DATA_ROOT}/containers" -maxdepth 1 -mindepth 1 2>/dev/null | wc -l)
                if [ "${n}" -gt 0 ]; then
                    log "Clearing ${n} stale container states before dockerd restart"
                    rm -rf "${DOCKER_DATA_ROOT}/containers"/* 2>/dev/null || true
                fi
            fi
            rm -f "${DOCKER_DATA_ROOT}/network/files/local-kv.db" 2>/dev/null || true
        else
            log "containerd-shim still alive, preserving container state"
        fi
    else
        log "Not in host PID namespace; cannot reliably enumerate shim — skipping state cleanup"
    fi

    # 4) 启动 dockerd（直接 nohup，不走 systemd）
    if [ -f "${PROXY_ENV_FILE}" ]; then
        # shellcheck disable=SC1090
        set -a; . "${PROXY_ENV_FILE}"; set +a
        log "Loaded proxy env from ${PROXY_ENV_FILE} before dockerd restart"
    else
        export HTTP_PROXY="${PROXY_URL}" HTTPS_PROXY="${PROXY_URL}"
        export http_proxy="${PROXY_URL}" https_proxy="${PROXY_URL}"
        export NO_PROXY="${NO_PROXY_LIST}" no_proxy="${NO_PROXY_LIST}"
        log "Proxy env file missing; exported PROXY_URL for dockerd restart: ${PROXY_URL}"
    fi
    nohup dockerd --containerd=/run/containerd/containerd.sock \
        > /tmp/dockerd_watchdog_restart.log 2>&1 &
    local pid=$!
    log "Started dockerd PID=${pid}, waiting for API..."

    local i
    for i in $(seq 1 60); do
        if docker_alive; then
            log "Docker API ready after ${i} attempts (~$((i*5))s)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            log "ERROR: dockerd died during startup; tail log:"
            tail -20 /tmp/dockerd_watchdog_restart.log 2>/dev/null | sed 's/^/  /' || true
            return 1
        fi
        sleep 5
    done
    log "ERROR: Docker failed to start after 5 min. See /tmp/dockerd_watchdog_restart.log"
    return 1
}

# ── Main ─────────────────────────────────────────────────────────────
log "========================================"
log "Starting docker_watchdog_v2 PID=$$"
log "  MAX_RUNNING=${MAX_RUNNING_CONTAINERS}  HARD_KILL=${HARD_KILL_THRESHOLD}"
log "  health every ${HEALTH_CHECK_INTERVAL}s; cgroup every ${CGROUP_MONITOR_INTERVAL}s; proc every ${PROC_MONITOR_INTERVAL}s"
log "  docker-cli every ${DOCKER_CLI_CHECK_INTERVAL}s timeout=${DOCKER_CLI_TIMEOUT}s fail_trigger=${MAX_CONSECUTIVE_DOCKER_CLI_FAILS}; proxy every ${PROXY_CHECK_INTERVAL}s"
log "  pool every ${POOL_CHECK_INTERVAL}s; deep probe every ${DEEP_PROBE_INTERVAL}s"
log "  heartbeat every ${HEARTBEAT_INTERVAL}s"
log "  disk every ${DISK_CHECK_INTERVAL}s; warn=${DISK_WARN_PCT}% emerg=${DISK_EMERGENCY_PCT}% min_free=${DISK_MIN_FREE_GB}GB inode_warn=${DISK_INODE_WARN_PCT}% inode_emerg=${DISK_INODE_EMERGENCY_PCT}%"
log "  pool_stop_on_disk_emergency=${POOL_STOP_ON_DISK_EMERGENCY}"
log "  PIDs warn=${PIDS_WARN_PCT}% emerg=${PIDS_EMERGENCY_PCT}%"
log "  proc warn: docker_related=${DOCKER_PROC_WARN} shim=${SHIM_PROC_WARN} runc=${RUNC_PROC_WARN} zombies=${ZOMBIE_WARN}"
log "  proc emerg: docker_related=${DOCKER_PROC_EMERGENCY} shim=${SHIM_PROC_EMERGENCY} runc=${RUNC_PROC_EMERGENCY} zombies=${ZOMBIE_EMERGENCY}"
log "  docker_down_shim_relief=${DOCKER_DOWN_SHIM_RELIEF} kill_shims_on_docker_down=${WATCHDOG_KILL_SHIMS_ON_DOCKER_DOWN}"
log "  Mem  warn=${MEM_WARN_PCT}% emerg=${MEM_EMERGENCY_PCT}%"
log "  pool=${POOL_HOST}:${POOL_PORT}  pool_server_regex=${POOL_SERVER_NAME_REGEX}"
log "  pool_pending repair=${POOL_PENDING_CLOSES_REPAIR} warn=${POOL_PENDING_CLOSES_WARN} threshold=${POOL_PENDING_CLOSES_REPAIR_THRESHOLD} stuck_checks=${POOL_PENDING_CLOSES_STUCK_CHECKS} active_max=${POOL_PENDING_CLOSES_ACTIVE_MAX} reap_limit=${POOL_PENDING_CLOSES_REAP_LIMIT} cooldown=${POOL_PENDING_CLOSES_REPAIR_COOLDOWN_S}s cancel_api=${POOL_PENDING_CLOSES_CANCEL_API} cancel_timeout=${POOL_PENDING_CLOSES_CANCEL_TIMEOUT}s"
log "  task_container_regex=${TASK_CONTAINER_REGEX}"
log "  task_image_regex=${TASK_IMAGE_REGEX}"
log "  docker_data_root=${DOCKER_DATA_ROOT}  proxy_url=${PROXY_URL}  proxy_env_file=${PROXY_ENV_FILE}"
log "  auto_repair=${WATCHDOG_AUTO_REPAIR} repair_mode=${WATCHDOG_REPAIR_MODE} repair_cooldown=${REPAIR_COOLDOWN_S}s repair_lock=${REPAIR_LOCK_DIR}"
log "  log_file=${LOG_FILE}  log_max=${LOG_MAX_BYTES}"

detect_pid_namespace
log "  pid_namespace: $([ "$HOST_PID_NS" = "1" ] && echo host || echo containerized)"

if detect_cgroup; then
    log "  cgroup: ${CGROUP_VERSION}"
    [ -n "$CGROUP_PIDS_DIR" ] && log "    pids: ${CGROUP_PIDS_DIR}  (max=${CGROUP_PIDS_MAX_VAL})"
    [ -n "$CGROUP_MEM_DIR"  ] && log "    mem : ${CGROUP_MEM_DIR}  (max=${CGROUP_MEM_MAX_VAL})"
else
    log "  cgroup: <NOT DETECTED — cgroup pressure disabled; /proc pressure still enabled>"
fi
log "========================================"

LAST_CLEANUP=0
LAST_CGROUP_CHECK=0
LAST_PROC_CHECK=0
LAST_DOCKER_CLI_CHECK=0
LAST_PROXY_CHECK=0
LAST_POOL_CHECK=0
LAST_DISK_CHECK=0
LAST_HEARTBEAT_TS=0
HEALTH_FAILS=0
DOCKER_CLI_FAILS=0
DEEP_PROBE_FAILS=0
LAST_DOCKER_CLI_STATUS="?"
LAST_PROXY_STATUS="?"

while true; do
    NOW=$(date +%s)

    # 1) 浅探活
    if docker_alive; then
        HEALTH_FAILS=0
    else
        HEALTH_FAILS=$((HEALTH_FAILS + 1))
        log "Health check failed (${HEALTH_FAILS}/${MAX_CONSECUTIVE_HEALTH_FAILS})"
        if [ "${HEALTH_FAILS}" -ge "${MAX_CONSECUTIVE_HEALTH_FAILS}" ]; then
            trigger_repair "dockerd unix-socket ping failed ${HEALTH_FAILS} consecutive times" 1
            HEALTH_FAILS=0
            sleep 10
            continue
        fi
        sleep "${HEALTH_CHECK_INTERVAL}"
        continue
    fi

    # 2) 深度探活（5 min 一次）—— 抓 address-pool 耗尽这种 ping OK 但 reset 500 的形态
    if [ $((NOW - LAST_DEEP_PROBE_TS)) -ge "${DEEP_PROBE_INTERVAL}" ]; then
        if docker_deep_alive; then
            LAST_DEEP_PROBE_TS="$NOW"
            DEEP_PROBE_FAILS=0
        else
            DEEP_PROBE_FAILS=$((DEEP_PROBE_FAILS + 1))
            log "WARN: deep probe failed (network create/rm, fails=${DEEP_PROBE_FAILS}) — likely address-pool exhausted or docker CLI/API wedged"
            timeout 30 docker network prune -f >/dev/null 2>&1 || true
            if [ "${DEEP_PROBE_FAILS}" -ge 2 ]; then
                trigger_repair "deep docker network probe failed ${DEEP_PROBE_FAILS} consecutive times"
                DEEP_PROBE_FAILS=0
            fi
            LAST_DEEP_PROBE_TS="$NOW"
        fi
    fi

    # 3) /proc 进程压力监控：不依赖 docker CLI，尽量在 fork 失败前预警/修复
    if [ $((NOW - LAST_PROC_CHECK)) -ge "${PROC_MONITOR_INTERVAL}" ]; then
        monitor_proc_pressure
        LAST_PROC_CHECK="$NOW"
    fi

    # 4) cgroup 监控
    if [ $((NOW - LAST_CGROUP_CHECK)) -ge "${CGROUP_MONITOR_INTERVAL}" ]; then
        monitor_pod_cgroup
        LAST_CGROUP_CHECK="$NOW"
    fi

    # 5) Docker CLI 探针：dockerd _ping OK 但 CLI/daemon metadata 卡死时触发
    if [ $((NOW - LAST_DOCKER_CLI_CHECK)) -ge "${DOCKER_CLI_CHECK_INTERVAL}" ]; then
        monitor_docker_cli
        LAST_DOCKER_CLI_CHECK="$NOW"
    fi

    # 6) proxy 低频探测：只告警，restart_docker 会显式带上 PROXY_URL
    if [ $((NOW - LAST_PROXY_CHECK)) -ge "${PROXY_CHECK_INTERVAL}" ]; then
        monitor_proxy
        LAST_PROXY_CHECK="$NOW"
    fi

    # 7) pool_server 监控
    if [ $((NOW - LAST_POOL_CHECK)) -ge "${POOL_CHECK_INTERVAL}" ]; then
        check_pool_server || true
        LAST_POOL_CHECK="$NOW"
    fi

    # 8) 容器清理 + 上限
    if [ $((NOW - LAST_CLEANUP)) -ge "${CLEANUP_INTERVAL}" ]; then
        cleanup_stopped
        enforce_container_limit
        LAST_CLEANUP="$NOW"
    fi

    # 9) Docker data-root 磁盘压力监控
    if [ $((NOW - LAST_DISK_CHECK)) -ge "${DISK_CHECK_INTERVAL}" ]; then
        monitor_docker_disk
        LAST_DISK_CHECK="$NOW"
    fi

    # 10) 低频心跳（默认 10 min）—— 复用上面已采集的指标，不发起新的 docker / curl
    if [ $((NOW - LAST_HEARTBEAT_TS)) -ge "${HEARTBEAT_INTERVAL}" ]; then
        log "OK: dockerd alive | docker_cli=${LAST_DOCKER_CLI_STATUS} proxy=${LAST_PROXY_STATUS} | pids=${LAST_PIDS_CUR:-?}/${LAST_PIDS_MAX:-?} (${LAST_PIDS_PCT:-?}%) tasks=${LAST_PROC_TASKS:-?} zombies=${LAST_ZOMBIES:-?} shim=${LAST_SHIM_PROCS:-?} runc=${LAST_RUNC_PROCS:-?} | pool active=${LAST_POOL_ACTIVE} pending_closes=${LAST_POOL_PENDING} | bridges=${LAST_BRIDGE_NETS} | task_containers=${LAST_RUNNING_TASKS}"
        LAST_HEARTBEAT_TS="$NOW"
    fi

    sleep "${HEALTH_CHECK_INTERVAL}"
done
