#!/usr/bin/env bash
# Fix docker compose V2 + proxy on CPU worker.
# Run on the CPU/docker worker:
#   bash terminal-rl/remote/fix_docker_cpu.sh
set -euo pipefail

echo "=== Step 1: Install docker compose V2 plugin ==="
DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}"
mkdir -p "$DOCKER_CONFIG/cli-plugins"

COMPOSE_URL="https://github.com/docker/compose/releases/download/v2.29.1/docker-compose-linux-x86_64"
PROXY="http://httpproxy-headless.kubebrain.svc.pjlab.local:3128"

if [ -x "$DOCKER_CONFIG/cli-plugins/docker-compose" ]; then
  echo "  docker-compose plugin already exists, skipping download"
else
  echo "  Downloading docker-compose v2.29.1 (via proxy)..."
  curl -SL --proxy "$PROXY" "$COMPOSE_URL" -o "$DOCKER_CONFIG/cli-plugins/docker-compose"
  chmod +x "$DOCKER_CONFIG/cli-plugins/docker-compose"
fi

echo "  Verifying..."
docker compose version || { echo "FAIL: docker compose not working"; exit 1; }
echo ""

echo "=== Step 2: Configure docker daemon proxy ==="
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/proxy.conf > /dev/null <<EOF
[Service]
Environment="HTTP_PROXY=$PROXY"
Environment="HTTPS_PROXY=$PROXY"
Environment="NO_PROXY=localhost,127.0.0.1,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn"
EOF
echo "  Proxy config written to /etc/systemd/system/docker.service.d/proxy.conf"

echo "  Reloading systemd and restarting docker..."
sudo systemctl daemon-reload
sudo systemctl restart docker
sleep 2

echo "  Verifying docker proxy..."
docker info 2>&1 | grep -i proxy || echo "  (no proxy line in docker info, but may still work)"
echo ""

echo "=== Step 3: Test build a sample task ==="
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL
export DATASET_DIR="terminal-rl/dataset"
echo "  Building task 996..."
docker compose -p build_run -f terminal-rl/dataset/seta_env/996/docker-compose.yaml build 2>&1 | tail -10
BUILD_RC=${PIPESTATUS[0]}

if [ "$BUILD_RC" -eq 0 ]; then
  echo ""
  echo "=== SUCCESS: docker compose build works ==="
  echo "  Pool server will now be able to build task containers."
  echo "  Restart pool server with:"
  echo "    pkill -f 'terminal-rl.remote.pool_server' || true"
  echo "    bash terminal-rl/remote/run_pool_server_pu.sh"
else
  echo ""
  echo "=== FAILED (exit $BUILD_RC) ==="
  echo "  Check docker logs: journalctl -u docker --no-pager -n 30"
fi
