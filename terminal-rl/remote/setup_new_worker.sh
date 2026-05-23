#!/usr/bin/env bash
# One-shot setup for a NEW CPU worker to run terminal-rl pool_server.
#
# Requirements for the new machine:
#   - Linux (Ubuntu 20.04+)
#   - Docker installed (or this script will try to install it)
#   - Network access (direct or via proxy)
#   - At least 200GB free disk on the partition where docker stores data
#   - Can reach GPU worker on port 18081 (pool_server listen port)
#
# Usage:
#   bash terminal-rl/remote/setup_new_worker.sh
#
# Environment variables (optional):
#   PROXY_URL    - HTTP proxy for docker daemon + pip (auto-detected if setup_proxy.sh available)
#   DOCKER_ROOT  - Custom docker data root if /var/lib/docker is too small
#                  (e.g. /mnt/large-disk/docker)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

echo "============================================================"
echo " Terminal-RL Pool Server - New Worker Setup"
echo " $(date)"
echo "============================================================"
echo ""

# ── 0. Detect proxy ─────────────────────────────────────────────────
PROXY_URL="${PROXY_URL:-}"
if [ -z "$PROXY_URL" ]; then
  # Try the pjlab proxy setup script
  if curl -fsS --max-time 3 "http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh" >/dev/null 2>&1; then
    PROXY_URL="http://httpproxy-headless.kubebrain.svc.pjlab.local:3128"
    echo "[auto] Detected pjlab proxy: $PROXY_URL"
  fi
fi
if [ -n "$PROXY_URL" ]; then
  export http_proxy="$PROXY_URL"
  export https_proxy="$PROXY_URL"
  export HTTP_PROXY="$PROXY_URL"
  export HTTPS_PROXY="$PROXY_URL"
  export no_proxy="localhost,127.0.0.1,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn"
  export NO_PROXY="$no_proxy"
fi

# ── 1. Check disk space ─────────────────────────────────────────────
echo "=== 1. Disk Space Check ==="
DOCKER_ROOT="${DOCKER_ROOT:-/var/lib/docker}"
DOCKER_PARTITION=$(df "$DOCKER_ROOT" --output=target 2>/dev/null | tail -1)
AVAIL_GB=$(df -BG --output=avail "$DOCKER_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')

echo "  Docker data root: $DOCKER_ROOT"
echo "  Partition: $DOCKER_PARTITION"
echo "  Available: ${AVAIL_GB}GB"
echo ""

if [ "${AVAIL_GB:-0}" -lt 150 ]; then
  echo "  [WARN] Less than 150GB available. Full training needs ~100-200GB for docker images."
  echo "         Consider setting DOCKER_ROOT to a larger partition."
  echo ""
  echo "  Available mount points with >150GB:"
  df -BG --output=target,avail 2>/dev/null | awk 'NR>1 && $2+0 > 150 {print "    "$1" ("$2" free)"}'
  echo ""
  read -p "  Continue anyway? [y/N] " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "  Aborted. Set DOCKER_ROOT=/path/to/large/disk and re-run."
    exit 1
  fi
fi

# ── 2. Install Docker (if missing) ──────────────────────────────────
echo "=== 2. Docker Installation ==="
if command -v docker &>/dev/null; then
  echo "  [OK] Docker already installed: $(docker --version)"
else
  echo "  Installing docker..."
  sudo apt-get update
  sudo apt-get install -y docker.io
  sudo systemctl enable docker
  sudo systemctl start docker
  echo "  [OK] Docker installed"
fi

# Ensure current user can use docker
if ! docker info >/dev/null 2>&1; then
  sudo usermod -aG docker "$USER" 2>/dev/null || true
  echo "  [NOTE] Added $USER to docker group. You may need to re-login or run: newgrp docker"
fi

# ── 3. Install Docker Compose V2 ────────────────────────────────────
echo "=== 3. Docker Compose V2 ==="
if docker compose version >/dev/null 2>&1; then
  echo "  [OK] $(docker compose version)"
else
  echo "  Installing docker compose V2 plugin..."
  DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}"
  mkdir -p "$DOCKER_CONFIG/cli-plugins"
  COMPOSE_URL="https://github.com/docker/compose/releases/download/v2.29.1/docker-compose-linux-x86_64"
  if [ -n "$PROXY_URL" ]; then
    curl -SL --proxy "$PROXY_URL" "$COMPOSE_URL" -o "$DOCKER_CONFIG/cli-plugins/docker-compose"
  else
    curl -SL "$COMPOSE_URL" -o "$DOCKER_CONFIG/cli-plugins/docker-compose"
  fi
  chmod +x "$DOCKER_CONFIG/cli-plugins/docker-compose"
  echo "  [OK] $(docker compose version)"
fi

# ── 4. Configure Docker daemon (proxy + data-root + address pools) ──
echo "=== 4. Docker Daemon Configuration ==="

# 4a. Proxy for pulling images
if [ -n "$PROXY_URL" ]; then
  sudo mkdir -p /etc/systemd/system/docker.service.d
  sudo tee /etc/systemd/system/docker.service.d/proxy.conf > /dev/null <<EOF
[Service]
Environment="HTTP_PROXY=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128"
Environment="HTTPS_PROXY=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128"
Environment="NO_PROXY=10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn"
EOF
  echo "  [OK] Docker daemon proxy configured"
fi

# 4b. daemon.json (data-root + address pools)
DAEMON_JSON="/etc/docker/daemon.json"
if [ "$DOCKER_ROOT" != "/var/lib/docker" ]; then
  sudo mkdir -p "$DOCKER_ROOT"
  sudo tee "$DAEMON_JSON" > /dev/null <<EOF
{
"registry-mirrors":[
"https://docker.1ms.run",
"https://docker.m.daocloud.io",
"https://dockerproxy.com",
"https://mirror.ccs.tencentyun.com"
],
"insecure-registries": [
"registry.h.pjlab.org.cn"
],
  "data-root": "/data",
  "storage-driver": "overlay2",
  "live-restore": true,
  "max-concurrent-downloads": 6,
  "max-concurrent-uploads": 6,
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "3"
  },
  "default-address-pools": [
    {"base": "10.200.0.0/16", "size": 24}
  ],
  "default-ulimits": {
    "nproc": {
      "Name": "nproc",
      "Hard": 4096,
      "Soft": 2048
    },
    "nofile": {
      "Name": "nofile",
      "Hard": 65536,
      "Soft": 65536
    },
    "core": {
      "Name": "core",
      "Hard": 0,
      "Soft": 0
    }
  },
  "default-shm-size": "64M"

}

EOF
  echo "  [OK] Docker data-root set to $DOCKER_ROOT"
else
  # Just set address pools (prevent subnet exhaustion per issue #3)
  if [ ! -f "$DAEMON_JSON" ] || ! grep -q "default-address-pools" "$DAEMON_JSON" 2>/dev/null; then
    sudo tee "$DAEMON_JSON" > /dev/null <<EOF
{
  "default-address-pools": [
    {"base": "10.200.0.0/16", "size": 24}
  ]
}
EOF
    echo "  [OK] Docker address pools configured"
  else
    echo "  [OK] daemon.json already configured"
  fi
fi

# 4c. Restart docker
echo "  Restarting docker daemon..."
sudo systemctl daemon-reload
sudo systemctl restart docker
sleep 3
docker info >/dev/null 2>&1 || { echo "  [FAIL] Docker not responding after restart"; exit 1; }
echo "  [OK] Docker daemon running"
echo ""

# ── 5. Pre-pull base images ─────────────────────────────────────────
echo "=== 5. Pre-pull Base Images ==="
BASE_IMAGES=(
  "ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624"
  "ghcr.io/laude-institute/t-bench/python-3-13:20250620"
  "ubuntu:22.04"
)
for img in "${BASE_IMAGES[@]}"; do
  if docker image inspect "$img" >/dev/null 2>&1; then
    echo "  [cached] $img"
  else
    echo "  [pulling] $img ..."
    docker pull "$img" 2>&1 | grep -E "Pull complete|Digest|Status" | tail -3
  fi
done
echo ""

# ── 6. Python environment ───────────────────────────────────────────
echo "=== 6. Python Environment ==="
cd "$REPO_ROOT"
if [ -d ".venv" ] && [ -x ".venv/bin/python" ]; then
  echo "  [OK] .venv exists"
else
  echo "  Creating .venv..."
  if command -v uv &>/dev/null; then
    uv venv .venv --python 3.12
  else
    python3 -m venv .venv
  fi
fi
source .venv/bin/activate

# Install pool_server dependencies
echo "  Installing dependencies..."
pip install --quiet terminal-bench fastapi uvicorn camel-ai 2>&1 | tail -3 || \
  pip install --quiet terminal-bench fastapi uvicorn camel-ai --no-deps 2>&1 | tail -3
echo "  [OK] Python deps installed"
echo ""

# ── 7. Verify build works ───────────────────────────────────────────
echo "=== 7. Verify Docker Build ==="
export DATASET_DIR="terminal-rl/dataset"
echo "  Building task 100 (simple task)..."
timeout 180 docker compose -p test_build \
  -f terminal-rl/dataset/seta_env/100/docker-compose.yaml build 2>&1 | tail -5
if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo "  [OK] Build succeeded"
  docker compose -p test_build -f terminal-rl/dataset/seta_env/100/docker-compose.yaml down 2>/dev/null || true
else
  echo "  [WARN] Build failed - check docker logs"
fi
echo ""

# ── 8. Summary ──────────────────────────────────────────────────────
MY_IP=$(hostname -I | awk '{print $1}')
echo "============================================================"
echo " Setup Complete!"
echo ""
echo " Machine IP: $MY_IP"
echo " Docker root: $(docker info 2>/dev/null | grep 'Docker Root Dir' | awk '{print $NF}')"
echo " Docker disk: $(df -h "$DOCKER_ROOT" --output=avail | tail -1 | xargs) available"
echo " Base images: pre-pulled"
echo " Compose V2: $(docker compose version 2>/dev/null | awk '{print $NF}')"
echo ""
echo " To start pool_server:"
echo "   cd $REPO_ROOT"
echo "   source .venv/bin/activate"
echo "   bash terminal-rl/remote/run_pool_server_pu_2.sh"
echo ""
echo " Then on GPU worker, set:"
echo "   export WORKER_URLS=\"http://${MY_IP}:18081\""
echo "   bash terminal-rl/terminal-rl_qwen3-8b_pu.sh"
echo "============================================================"
