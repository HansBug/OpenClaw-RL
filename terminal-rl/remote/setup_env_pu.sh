#!/usr/bin/env bash
# Pre-pull base images + assess system resources for terminal-rl training.
# Run on CPU worker:
#   bash terminal-rl/remote/setup_env_pu.sh
#
# This script:
#   1. Checks system resources (disk, docker, network)
#   2. Pre-pulls the 3 base images used by seta_env tasks (~200MB total)
#   3. Configures docker auto-cleanup (prune dangling images when disk > 80%)
#   4. Restarts pool_server with the _pu wrapper (realtime logs + file)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

echo "============================================================"
echo " Terminal-RL CPU Worker Setup"
echo " $(date)"
echo "============================================================"
echo ""

# ── 1. System resource check ────────────────────────────────────────
echo "=== 1. System Resources ==="
echo "--- Disk ---"
df -h / /var/lib/docker 2>/dev/null | head -5
echo ""
echo "--- Docker ---"
docker info 2>&1 | grep -E "Server Version|Storage Driver|Docker Root Dir|Total Memory" || true
docker system df 2>/dev/null
echo ""
echo "--- Docker proxy ---"
docker info 2>&1 | grep -iE "proxy|mirror" || echo "  (no proxy configured in daemon)"
echo ""

# ── 2. Pre-pull base images ─────────────────────────────────────────
echo "=== 2. Pre-pulling base images (one-time, ~200MB total) ==="
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
    docker pull "$img" 2>&1 | tail -3
  fi
done
echo ""

# ── 3. Verify docker compose works ─────────────────────────────────
echo "=== 3. Verify docker compose build ==="
docker compose version || { echo "FAIL: docker compose V2 not installed"; exit 1; }

# Quick build test (task 996, should be fast now that base is cached)
echo "  Test building task 996..."
cd "$REPO_ROOT"
export DATASET_DIR="terminal-rl/dataset"
timeout 120 docker compose -p build_run \
  -f terminal-rl/dataset/seta_env/996/docker-compose.yaml build 2>&1 | tail -5
BUILD_RC=${PIPESTATUS[0]}
if [ "$BUILD_RC" -eq 0 ]; then
  echo "  [OK] task 996 build succeeded"
else
  echo "  [WARN] task 996 build failed (rc=$BUILD_RC), check Dockerfile"
fi
echo ""

# ── 4. Configure docker auto-prune (prevent disk full) ──────────────
echo "=== 4. Docker auto-prune setup ==="
# Since all 1357 tasks will eventually be sampled, we want to keep built
# images as long as possible (they serve as cache). Only prune when disk
# is critically full (>90%), and only remove dangling/intermediate layers,
# NOT named task images.
PRUNE_SCRIPT="/usr/local/bin/docker-auto-prune.sh"
sudo tee "$PRUNE_SCRIPT" > /dev/null <<'PRUNE'
#!/bin/bash
USAGE=$(df /var/lib/docker --output=pcent | tail -1 | tr -dc '0-9')
if [ "${USAGE:-0}" -gt 90 ]; then
  echo "[$(date)] Disk at ${USAGE}%, pruning dangling images and build cache..."
  docker image prune -f 2>&1 | tail -3
  docker builder prune -f --filter "until=1h" 2>&1 | tail -3
fi
PRUNE
sudo chmod +x "$PRUNE_SCRIPT"

# Add to crontab if not already there
if ! crontab -l 2>/dev/null | grep -q "docker-auto-prune"; then
  (crontab -l 2>/dev/null; echo "*/10 * * * * $PRUNE_SCRIPT >> /tmp/docker-prune.log 2>&1") | crontab -
  echo "  [OK] Auto-prune cron installed (every 10 min, triggers at >80% disk)"
else
  echo "  [OK] Auto-prune cron already exists"
fi
echo ""

# ── 5. Pool server status ───────────────────────────────────────────
echo "=== 5. Pool server ==="
if pgrep -af "terminal-rl.remote.pool_server" >/dev/null 2>&1; then
  echo "  [running] pool_server is already up"
  curl -fsS --max-time 3 http://127.0.0.1:18081/healthz && echo " OK" || echo " UNHEALTHY"
else
  echo "  [stopped] pool_server not running"
  echo ""
  echo "  To start (with realtime terminal output + log file):"
  echo "    cd $REPO_ROOT"
  echo "    bash terminal-rl/remote/run_pool_server_pu.sh"
fi
echo ""

# ── 6. Summary ──────────────────────────────────────────────────────
echo "============================================================"
echo " Setup complete. Summary:"
echo ""
echo " Base images: pre-pulled (cached)"
echo " Build strategy: on-demand (pool_server builds on first /reset)"
echo "   - 1357 tasks total, all will be used during full training"
echo "   - First ~170 rollouts: each batch builds ~8 new tasks (1-3 min each)"
echo "   - After ~170 rollouts: all tasks cached, builds are instant"
echo "   - Estimated warm-up phase: ~30-60 min of extra build time"
echo " Auto-prune: enabled (>90% disk triggers dangling-only cleanup)"
echo " Disk headroom: $(df -h /var/lib/docker --output=avail | tail -1 | xargs) available"
echo ""
echo " Next steps:"
echo "   1. Start pool_server:  bash terminal-rl/remote/run_pool_server_pu.sh"
echo "   2. On GPU worker:      bash terminal-rl/terminal-rl_qwen3-8b_pu.sh"
echo "============================================================"
