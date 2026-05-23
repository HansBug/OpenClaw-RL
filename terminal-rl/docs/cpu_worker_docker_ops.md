# CPU Worker Docker 稳定运行手册

本文档汇总了 OpenClaw-RL 训练 CPU worker 上 docker / pool_server / docker-watchdog 长跑稳定性的运维经验。**14 h 长跑场景**下踩过的坑都在这里。

相关代码（active workflow）：
- `terminal-rl/remote/setup_new_worker.sh` — 从零配置：装 docker、配 daemon.json、预拉 base image
- `terminal-rl/remote/fix_dockerd_and_proxy.sh` — 一站式代理修复（watchdog-aware，4 层代理注入）
- `terminal-rl/remote/prebuild_proxied_base_images.sh` — base image apt 代理注入（fix 脚本会自动调）
- `terminal-rl/remote/run_pool_server_pu_v2.sh` — 加固版 pool_server 启动脚本
- `terminal-rl/remote/docker_watchdog_v2.sh` — docker 看门狗
- `terminal-rl/remote/docker-watchdog.service` — systemd unit
- `terminal-rl/remote/restart_docker_force.sh` — dockerd 强制重启（绕开 systemctl）
- `terminal-rl/remote/cleanup_docker_cache.sh` — 安全清理 build cache / dangling image
- `terminal-rl/remote/diag_docker_failures_lite.sh` — 训练并行可用的诊断快照

旧版脚本（`fix_build_proxy_full.sh` / `setup.sh` / `run_pool_server.sh` / 等）已迁至 `terminal-rl/archive/remote/`，仅供查阅。

相关 issue 与分析：
- [issue #3](https://github.com/HansBug/OpenClaw-RL/issues/3) — 6 个 env-pool 踩坑（坑1-6）
- `runs/terminal-rl_qwen3-8b_8gpu_2026-05-21_124958/metrics/analysis/REPORT.md` — 14h 崩溃案例分析

---

## 0. 从零配置 CPU worker（首次部署）

> 适用场景：拿到一台**全新**的 CPU worker，要从零开始让它能承接 GPU worker 的训练任务。整个流程约 **20–40 分钟**（取决于 base image 拉取速度）。

### 0.1 前置条件

| 项 | 要求 |
|---|---|
| OS | Ubuntu 20.04+ |
| 用户 | 能 `sudo`（脚本要写 `/etc/`、装 docker、改 systemd） |
| 磁盘 | docker data-root 所在分区 ≥ **150 GB**（推荐 ≥ 500 GB，14h 训练会累积大量 image/container layer） |
| 网络 | 能直连或经代理访问 `ghcr.io` / `docker.io` / `archive.ubuntu.com`；本环境是 pjlab 内网代理 `http://httpproxy-headless.kubebrain.svc.pjlab.local:3128` |
| 端口 | 18081 出方向能被 GPU worker 访问 |
| 仓库 | OpenClaw-RL 已 clone 到共享存储（默认 `/mnt/shared-storage-user/puyuan/code/OpenClaw-RL`） |

### 0.2 五步完成（按顺序执行）

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL

# ─── Step 1: 装 docker / compose / daemon.json / 预拉 base image ───
# 自动检测 pjlab 代理；data-root 设到 /data（≥500GB 大盘）
sudo DOCKER_ROOT=/data bash terminal-rl/remote/setup_new_worker.sh

# ─── Step 2: 一站式 4 层代理注入（含 base image apt 代理） ───
# Phase 1 停 watchdog（如果安装了）→ Phase 2 强重启 dockerd →
# Phase 3 写 4 处代理配置 → Phase 4 启动 dockerd →
# Phase 5 验证 dockerd env → Phase 5.5 wrap base images →
# Phase 6 验证 seta_env/0 build → Phase 7 重启 watchdog
sudo bash terminal-rl/remote/fix_dockerd_and_proxy.sh

# ─── Step 3: 部署 docker-watchdog（systemd 持久守护） ───
sudo cp terminal-rl/remote/docker-watchdog.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now docker-watchdog
journalctl -u docker-watchdog -n 20 --no-pager   # 确认看到 banner + 无 WARN

# ─── Step 4: Python 环境（如果 setup_new_worker.sh 已建好就跳过） ───
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install --quiet terminal-bench fastapi uvicorn camel-ai

# ─── Step 5: 验收 ───
docker info >/dev/null 2>&1 && echo "[OK] dockerd"
docker history ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624 \
  | head -3 | grep -q "Acquire::http" && echo "[OK] apt proxy wrap"
systemctl is-active docker-watchdog && echo "[OK] watchdog"
```

通过以上 5 步后，机器**不需要再做任何 docker 配置**，直接进入 §2 启动 pool_server 即可。

### 0.3 失败回退矩阵

| 失败发生在 | 检查 | 应对 |
|---|---|---|
| Step 1 "Restarting docker daemon" 卡住 | `systemctl status docker` | Ctrl-C 中断后**不要**自己 `systemctl restart docker`，直接进 Step 2，`fix_dockerd_and_proxy.sh` 会用 `restart_docker_force.sh` 路径恢复 |
| Step 2 Phase 0 "proxy not reachable" | `curl -x $PROXY_URL http://example.com` | DNS / 代理服务下线，先排网络 |
| Step 2 Phase 5.5 wrap 某个 base image 失败 | `docker pull <base>` 单独验证 | 多半是该 base image 在 ghcr/docker.io 上找不到；改 `BASE_IMAGES` env 跳过 |
| Step 2 Phase 6 build 仍 timeout | `docker run --rm <base> sh -c 'apt-get update'` | wrap 没生效，重跑 `prebuild_proxied_base_images.sh` |
| Step 3 watchdog 起不来 | `journalctl -u docker-watchdog -n 50 --no-pager` | 多半是 dropin 语法错误；`ls /etc/systemd/system/docker-watchdog.service.d/` 检查 |

### 0.4 配置产物清单（脚本写到了哪里）

| 文件 | 写入者 | 作用 |
|---|---|---|
| `/etc/docker/daemon.json` | setup_new_worker.sh | data-root / address-pool / nofile |
| `/etc/systemd/system/docker.service.d/proxy.conf` | setup_new_worker.sh | dockerd HTTP_PROXY |
| `/etc/systemd/system/docker.service.d/http-proxy.conf` | fix_dockerd_and_proxy.sh | 同上（覆盖 systemd 启动路径） |
| `/etc/systemd/system/docker-watchdog.service.d/http-proxy.conf` | fix_dockerd_and_proxy.sh | watchdog nohup 起的 dockerd 继承代理 |
| `/root/.docker/config.json` (+ puyuan/.docker/config.json) | fix_dockerd_and_proxy.sh | docker build 自动注入 HTTP_PROXY env |
| `/etc/seta_build_proxy.env` | fix_dockerd_and_proxy.sh | pool_server 启动时 `set -a; . file; set +a` 继承 |
| `/etc/systemd/system/docker-watchdog.service` | 手动 cp | watchdog systemd unit |

---

## 1. 整体拓扑与故障域

```
┌──── GPU Worker ─────────┐                ┌──── CPU Worker (share-machine) ───────┐
│                         │  HTTP /reset  │                                       │
│  slime trainer ─────────┼──────────────►│  pool_server :18081                   │
│  router :18080          │  /exec_tool   │    │ DooD: --network host              │
│                         │  /evaluate    │    │ mounts /var/run/docker.sock       │
│                         │  /close       │    ▼                                  │
└─────────────────────────┘                │  dockerd ──► task containers          │
                                           │             (per-task bridge nets)    │
                                           │                                       │
                                           │  docker-watchdog (systemd)            │
                                           │    └── 监控 dockerd + pool_server     │
                                           │        + cgroup pressure              │
                                           └───────────────────────────────────────┘
```

**故障域划分**：

| 层 | 故障形态 | 防御 |
|---|---|---|
| dockerd 完全死 | `/_ping` 超时 | watchdog 浅探活 + restart_docker |
| dockerd 半死 | `/_ping` OK 但 `/reset` 500 | watchdog 深探活（`network create+rm`） |
| address-pool 耗尽 | `compose up` 报 "all pools fully subnetted" | 扩 daemon.json + watchdog 网络数预警 |
| nofile=1024 | evaluate 阶段 "Too many open files" | nofile 永久抬到 65k+ |
| 长跑残留 | 孤儿容器 / 网络 → 新 reset 失败 | pool_server v2 pre-flight 清理 + watchdog 周期 prune |
| pool_server cgroup 压力 | PIDs / Memory 接近上限 | watchdog cgroup monitor + emergency relief |
| 内网代理下 build 全 exit-17 | `apt-get update` 连不上 archive.ubuntu.com | `prebuild_proxied_base_images.sh` 在 base image 写 `apt.conf.d/95proxies` |

---

## 2. 启动顺序（每次训练前）

> 适用场景：机器已按 §0 完成首次配置，现在要跑一次新训练。

### 2.1 Pre-flight（30 秒，确认基础设施健康）

```bash
# 在 CPU worker 上
docker info >/dev/null 2>&1 && echo "[OK] dockerd alive"
systemctl is-active docker-watchdog && echo "[OK] watchdog running"
test -f /etc/seta_build_proxy.env && echo "[OK] proxy env file exists"
docker history ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624 \
  | head -3 | grep -q "Acquire::http" && echo "[OK] base image apt proxy wrap"
ss -tlnp | grep -q ':18081 ' && echo "[WARN] port 18081 already bound (kill old pool_server)" \
  || echo "[OK] port 18081 free"
```

如果任何一项 `[FAIL]` / `[WARN]`：
- `dockerd alive` fail → §4.2 强重启
- `watchdog running` fail → `sudo systemctl start docker-watchdog`
- `proxy env file` 缺失 → 跑 §0 Step 2
- `apt proxy wrap` 缺失 → `sudo bash terminal-rl/remote/prebuild_proxied_base_images.sh`
- `port 18081` 已占 → `lsof -i :18081`，`kill -9 <PID>`

### 2.2 启动 pool_server

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL

# 关键：必须 source proxy env，否则 docker_compose_utils.py 拿不到 HTTP_PROXY
set -a; . /etc/seta_build_proxy.env; set +a

source .venv/bin/activate

# 前台跑（看实时日志，Ctrl-C 退出）
bash terminal-rl/remote/run_pool_server_pu_v2.sh

# 或后台跑（推荐用于生产 14h 长跑）
nohup bash terminal-rl/remote/run_pool_server_pu_v2.sh \
  > /tmp/cpu_pool.log 2>&1 &
echo $! > /tmp/cpu_pool.pid
```

### 2.3 验证（另开一个终端）

```bash
# pool_server health
curl --noproxy '*' http://127.0.0.1:18081/healthz
# 期望：{"ok": true, "active_tasks": 0, ...}

# pool_server status（容量配置）
curl --noproxy '*' http://127.0.0.1:18081/status | python3 -m json.tool

# watchdog 心跳（每 10 分钟一行）
journalctl -u docker-watchdog -n 5 --no-pager | grep "OK: dockerd alive"
```

### 2.4 通知 GPU worker

```bash
# 在 CPU worker 上拿到自己的 IP
hostname -I | awk '{print $1}'
# 假设是 10.103.5.46

# 然后在 GPU worker 上：
export WORKER_URLS="http://10.103.5.46:18081"
bash terminal-rl/terminal-rl_qwen3-8b_pu.sh
```

### 2.5 训练结束后清理（可选）

```bash
# 在 CPU worker 上
kill -9 $(cat /tmp/cpu_pool.pid) 2>/dev/null
rm /tmp/cpu_pool.pid

# 清理 build cache（保留运行中的容器和有 tag 的 image）
bash terminal-rl/remote/cleanup_docker_cache.sh
```

watchdog 不需要停，它会一直在后台监管 dockerd 健康。

---

## 3. docker-watchdog 工作机制

watchdog 由 systemd 拉起（`docker-watchdog.service`），每 30 s 跑一次主循环，按不同节奏检查多个维度：

| 检查 | 周期 | 触发条件 → 动作 |
|---|---|---|
| **浅探活** | 30 s | `dockerd /_ping` 连续失败 3 次 → `restart_docker`（pkill + nohup） |
| **深探活** | 5 min | `network create + rm` 失败 → `network prune` |
| **cgroup 压力** | 15 s | PIDs/Mem ≥90% → emergency kill 任务容器（60 s 冷却） |
| **pool_server 健康** | 30 s | `/healthz` 不通 → WARN；`pending_closes>50` → `network prune` |
| **bridge 网络数** | 30 s | `>200` → `network prune`（防 address-pool 耗尽） |
| **stopped 容器清理** | 60 s | `>5` 个 stopped → `container prune` |
| **运行容器上限** | 60 s | task 容器 `>80` 软杀 / `>120` 硬杀 |
| **心跳** | 10 min | 一行 `OK:` 状态摘要 |

### 关键日志关键词

```bash
journalctl -u docker-watchdog -f | grep -E "WARN|EMERGENCY|HARD LIMIT|Health check failed"
```

- `WARN: pool_server /healthz unreachable` — pool_server 崩了 / 端口变了 / 代理拦截
- `WARN: ${nets} bridge networks, address-pool risk` — 接近 address-pool 上限
- `WARN: deep probe failed` — dockerd 半死（`/_ping` 通但创建网络失败）
- `EMERGENCY: PIDs ${cur}/${max}` — cgroup 压力高，开始 kill 任务容器
- `HARD LIMIT: N task containers > 120` — task 容器超硬上限
- `Health check failed (3/3)` — 即将 restart_docker

### 心跳行示例

```
2026-05-22 14:05:06 [docker-watchdog] OK: dockerd alive | pool active=12 pending_closes=3 | bridges=87 | task_containers=24
```

字段含义：
- `pool active` — pool_server `/status` 中 `active_tasks`（含未释放的 lease）
- `pending_closes` — 排队等 docker compose down 的 lease 数；正常 < 10
- `bridges` — bridge 网络总数；接近 200 时风险上升
- `task_containers` — task 容器数（不含 pool_server 自己）

---

## 4. 常见故障与处置

### 4.1 pool_server `/reset` 大量 500

**症状**：训练日志 `Server error '500 Internal Server Error' for url 'http://x.x.x.x:18080/reset'` 短时间高频。

**先看 pool_server 日志定位根因**：

```bash
tail -100 /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/tmp_doc_latest/cpu_pool.log
```

| 关键字 | 根因 | 处置 |
|---|---|---|
| `all predefined address pools have been fully subnetted` | 坑3：address-pool 耗尽 | `docker network prune -f`；扩 daemon.json（见 §5） |
| `Too many open files` / `Errno 24` | 坑4：nofile=1024 | 永久抬 nofile（见 §5） |
| `i/o timeout` / `failed to fetch anonymous token` | 坑3.5：镜像源不可达 | 换 registry mirror；预拉基础镜像 |
| `name is already in use` / `endpoint already exists` | 坑5：孤儿容器/网络 | `docker container prune -f; docker network prune -f` |
| `TASK_SLOTS_EXHAUSTED` | 坑1：pool 容量不足 | 启动时调大 `--max-tasks` / `--max-runs-per-task` |
| `RUN_SLOTS_EXHAUSTED` | 坑1：单 task 并发不够 | `--max-runs-per-task >= n_samples_per_prompt` |

### 4.2 dockerd 完全无响应（`docker info` 卡死）

```bash
# 步骤：先观察 watchdog 是否在自动重启
journalctl -u docker-watchdog -n 50 --no-pager | grep -E "DOWN|restart"

# 如果 watchdog 也卡了 / 没装，手动强制重启
bash /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/terminal-rl/remote/restart_docker_force.sh
```

`restart_docker_force.sh` 设计要点：**完全绕开 systemctl restart**（D state 时会卡 D-Bus），用 `pkill -9 dockerd` + `stop docker.socket` + `nohup dockerd`。

### 4.3 watchdog `WARN: pool_server /healthz unreachable` 持续

按概率从高到低：

1. **HTTP 代理污染**（最常见）—— `http_proxy/HTTPS_PROXY` 让 curl 把 `127.0.0.1` 也走代理 → 502 Bad Gateway。修复：systemd unit 已加 `Environment=no_proxy=*` + 脚本 curl 用 `--noproxy '*'`。验证：`curl -v --noproxy '*' http://127.0.0.1:18081/healthz`
2. **pool_server 没起 / 在别的端口** —— `pgrep -af pool_server` + `ss -tlnp | grep 1808`；启动或 `Environment=POOL_PORT=xxx`
3. **pool_server 半死** —— 看 `cpu_pool.log` 最近输出；多半是 §4.1 的某种

### 4.4 watchdog 自身挂了 / 多实例并发

```bash
# 检查 systemd 实例数（应只 1）
pgrep -af docker_watchdog

# 期望只看到 1 行：
# NNNNNN /usr/bin/bash /mnt/.../docker_watchdog_v2.sh
```

**如果有多份**：通常是手动 `bash` 起过一份 + systemd 又起一份。立刻杀重复（保留 systemd 那个）：

```bash
sudo systemctl status docker-watchdog | grep "Main PID"  # 记下 PID
pgrep -af docker_watchdog | grep -v <SYSTEMD_PID> | awk '{print $1}' | xargs -r sudo kill -9
```

### 4.5 训练到 ~13 h 后 `/reset` 大量 500（05-21 案例）

这是 issue #3 坑5 完整复现 —— dockerd 长跑后状态污染。**有 watchdog v2 应已自动处理**。如果没装：

```bash
# 停训练 → 强制重启 docker → 重启 pool_server
bash terminal-rl/remote/restart_docker_force.sh
bash terminal-rl/remote/run_pool_server_pu_v2.sh
```

### 4.6 内网代理环境下 build 全部 exit-17（5-23 案例）

**症状**：`docker compose build` 失败，build 日志含 `Could not connect to archive.ubuntu.com:80 ... connection timed out`。1377 个 seta_env Dockerfile 中只有 2 个（`*_my/Dockerfile`）能 build，其余 99.8% 全部 exit-17。

**根因（精确定位）**：

`fix_dockerd_and_proxy.sh` 此前注入了 3 处代理却仍不够：
- ✅ `/etc/systemd/system/docker.service.d/http-proxy.conf` (dockerd 拉 base image 走代理)
- ✅ `/etc/systemd/system/docker-watchdog.service.d/http-proxy.conf` (watchdog nohup 起的 dockerd 也走代理)
- ✅ `~/.docker/config.json` 的 `proxies` 段（自动把 `HTTP_PROXY` 注入 build env）
- ❌ **APT 不读 `HTTP_PROXY` 环境变量** ← 缺这一环

**Ubuntu 24.04 的 apt 默认只读 `/etc/apt/apt.conf.d/*` 里的 `Acquire::http::Proxy` 指令，不读 env vars**。这是 Debian/Ubuntu 历史设计选择（避免 shell env 影响包源签名校验）。

`seta_env/0_my/Dockerfile` 之所以工作，是因为它显式做了：

```dockerfile
RUN if [ -n "$HTTP_PROXY" ]; then \
    echo 'Acquire::http::Proxy "'$HTTP_PROXY'";'  > /etc/apt/apt.conf.d/95proxies && \
    echo 'Acquire::https::Proxy "'$HTTPS_PROXY'";' >> /etc/apt/apt.conf.d/95proxies; \
fi
```

**修复**：不能改 1377 个 Dockerfile，最干净的解法是 **本地 wrap base image**：

```bash
# 一次性预构建带 apt 代理的 shadow base image（同 tag 覆盖 upstream）
sudo bash terminal-rl/remote/prebuild_proxied_base_images.sh
# 或 fix_dockerd_and_proxy.sh 已在 Phase 5.5 自动调用
```

`prebuild_proxied_base_images.sh` 行为：
1. 对前 4 个高频 base image（`ghcr.io/.../ubuntu-24-04:20250624` 1317 次、`ubuntu:22.04` 45 次、`ghcr.io/.../python-3-13:20250620` 8 次、`ubuntu:24.04` 4 次）
2. 各 build 一层薄包装：`FROM <base>` + `RUN echo Acquire::... > /etc/apt/apt.conf.d/95proxies`
3. 用 **同一个 tag** 覆盖 upstream（docker 优先用本地 image，所有 `FROM <tag>` 自动继承）
4. 烟雾测试：`docker run --rm <base> apt-get update`，看是否走代理成功

**坑**：

- 如果有人 `docker pull ghcr.io/...:20250624`，shadow image 会被 upstream 覆盖。**重跑** prebuild 即可恢复。
- 验证 wrap 是否生效：`docker history <base> | head -5` 顶层应该有 `RUN if [ -n "$HTTP_PROXY"...` 这一层。
- 与 issue #3 的 BuildKit frontend exit-17 是**两个不同的 exit-17**：
  - issue #3 §5：BuildKit `docker/dockerfile:1` frontend manifest pull 超时（已用 `DOCKER_BUILDKIT=0` 绕开）
  - 本节：legacy builder 模式下 apt 没代理 → connection timeout → compose 把 `apt-get` 的 exit 1 翻译为 exit 17

**SOP**：

```bash
# 一站式（推荐）
sudo bash terminal-rl/remote/fix_dockerd_and_proxy.sh   # Phase 5.5 自动 wrap base image

# 验证 4 个高频 base image 的 apt-get update 在 build 内能走代理
docker run --rm ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624 \
  sh -c 'apt-get update >/dev/null 2>&1 && echo OK'
```

---

## 5. 长期稳定的系统级配置（一次性）

### 5.1 扩大 docker default-address-pools（坑3 根治）

默认仅 256 个 /24 子网，64 并发就可能撑爆。改 `/etc/docker/daemon.json`：

```json
{
  "default-address-pools": [
    {"base": "10.200.0.0/12", "size": 24},
    {"base": "172.30.0.0/14", "size": 24}
  ],
  "data-root": "/data",
  "registry-mirrors": ["https://docker.1ms.run"]
}
```

dockerd 会规整为 `10.192.0.0/12 + 172.28.0.0/14`，约 **4096 个 /24 bridge 网络**。

生效需要：`sudo systemctl restart docker`（无任务运行时；watchdog 跟着重新连）。

### 5.2 永久抬 nofile（坑4 根治）

两处都要改：

```bash
# /etc/security/limits.conf
* soft nofile 65536
* hard nofile 65536
root soft nofile 65536
root hard nofile 65536

# /etc/systemd/system/docker.service.d/override.conf
[Service]
LimitNOFILE=65536
```

生效：`sudo systemctl daemon-reload && sudo systemctl restart docker`。

验证：`sudo cat /proc/$(pgrep -x dockerd)/limits | grep "open files"`，soft/hard 都应 ≥ 65536。

### 5.3 docker-watchdog systemd 部署

```bash
sudo cp /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/terminal-rl/remote/docker-watchdog.service \
        /etc/systemd/system/docker-watchdog.service
sudo systemctl daemon-reload
sudo systemctl enable --now docker-watchdog
journalctl -u docker-watchdog -f                       # 看 banner
```

常用 override：

```bash
sudo systemctl edit docker-watchdog
# 加：
[Service]
Environment=POOL_PORT=18081
Environment=MAX_RUNNING_CONTAINERS=80
Environment=BRIDGE_NETS_WARN=200
Environment=HEARTBEAT_INTERVAL=600
```

保存后：`sudo systemctl restart docker-watchdog`。

---

## 6. pool_server v2 容量配置参考

`run_pool_server_pu_v2.sh` 默认参数：

| 参数 | 默认 | 含义 | 调整建议 |
|---|---:|---|---|
| `WORKER_MAX_TASKS` | 64 | 同时占用的不同 task_key 上限 | ≥ rollout_batch_size |
| `WORKER_MAX_RUNS_PER_TASK` | 16 | 单 task 并发 lease 上限 | ≥ n_samples_per_prompt |
| `WORKER_MAX_CONCURRENT_CLOSES` | 32 | 同时 docker compose down 数 | ≈ 1.5 × rollout_batch_size |
| `ENV_SERVER_PORT` | 18081 | listen 端口 | 改要同步 watchdog `POOL_PORT` |
| `SKIP_PREFLIGHT_CLEANUP` | 0 | 启动前清孤儿 | 短时重启可设 1 |

**校准公式**（issue #3 §1.2）：
```
训练 demand = rollout_batch_size × n_samples_per_prompt
→ WORKER_MAX_TASKS × WORKER_MAX_RUNS_PER_TASK ≥ demand × 8 (推荐 8× headroom)
→ WORKER_MAX_RUNS_PER_TASK ≥ n_samples_per_prompt (硬要求)
```

以 8B 训练 `rollout_batch_size=16, n_samples=8` 为例：
- demand = 128
- 默认 64 × 16 = 1024 slots ≈ 8× headroom ✅
- runs_per_task=16 ≥ n_samples=8 ✅

---

## 7. 常用诊断命令

```bash
# 一行 dashboard（每 30 s 刷新）
watch -n 30 'echo "=== $(date) ==="; \
    echo "watchdog: $(systemctl is-active docker-watchdog)"; \
    echo "docker  : $(docker info >/dev/null 2>&1 && echo OK || echo DOWN)"; \
    echo "pool    : $(curl -fsS --max-time 2 --noproxy "*" http://127.0.0.1:18081/healthz >/dev/null 2>&1 && echo OK || echo DOWN)"; \
    echo "running : $(docker ps -q | wc -l) containers, $(docker network ls --filter driver=bridge -q | wc -l) bridges"'

# 看 pool_server 最近的 /reset 错误率
grep -aE "POST /reset HTTP" /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/tmp_doc_latest/cpu_pool.log \
    | tail -200 | awk -F'" ' '{print $2}' | awk '{print $1}' | sort | uniq -c
# 期望大量 200，少量 500

# 看 watchdog 心跳是否在按时输出（10 min 一次）
journalctl -u docker-watchdog --since "1 hour ago" --no-pager | grep "OK: dockerd alive"

# 看 cgroup pressure（如果 watchdog 报告未检测到 cgroup，自己手动看）
for f in /sys/fs/cgroup/pids/pids.{current,max} /sys/fs/cgroup/memory/memory.{usage_in_bytes,limit_in_bytes}; do
    [ -r "$f" ] && echo "$f: $(cat $f)"
done
```

---

## 8. 已知不足 & TODO

- **没监控 ClawSentry gateway** — 只监控 pool_server。如果 ClawSentry 限流（05-21 出现 2780×429），watchdog 不会自动处理
- **address-pool 扩容未自动化** — `/etc/docker/daemon.json` 还需手动配，建议加进 `setup_new_worker.sh`
- **没有自动 wandb 告警** — 长跑期间 `/reset 500` 暴增时只能看 log，没有自动通知
- **pool_server 自身没 self-watchdog** —— 当前依赖 watchdog 外部观察；pool_server 死了 watchdog 只能 log WARN，不会自动重启它。建议未来给 pool_server 也加 systemd unit

## 附录：重启 dockerd 的三档方案

| 方案 | 何时用 | 风险 |
|---|---|---|
| `systemctl restart docker` | dockerd 健康但需要重读配置 | dockerd D state 时会卡 D-Bus 数小时 |
| `bash restart_docker_force.sh` | dockerd 死了 / 假死 | pkill 后短暂无 docker 服务（30 s ~ 5 min） |
| `watchdog 自动 restart_docker` | watchdog 检测到 3×浅探活失败 | 同上，但有 60 s 冷却避免抖动 |

所有 "force" 方案都会**绕过 systemctl restart**，直接 pkill + nohup，并先 `systemctl stop docker.socket` 阻断 systemd 的 `Restart=always` race。
