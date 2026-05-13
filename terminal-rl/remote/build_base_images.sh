#!/usr/bin/env bash
# Build local replacements for ghcr.io base images that are unreachable.
#
# Problem: ghcr.io (20.205.243.164:443) is blocked by the corporate proxy.
# Solution: Build equivalent images locally from Docker Hub base + tmux/asciinema,
#           then tag them with the ghcr.io name so all task Dockerfiles work unchanged.
#
# Usage (on CPU worker):
#   bash terminal-rl/remote/build_base_images.sh
set -euo pipefail

echo "=== Building local base images (ghcr.io is unreachable) ==="
echo ""

# ── 1. ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624 ────────
# This is just ubuntu:24.04 + tmux + asciinema (the terminal-bench boilerplate)
echo "[1/2] Building t-bench/ubuntu-24-04 equivalent..."
docker build -t ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624 -f - . <<'DOCKERFILE'
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      tmux \
      asciinema \
      bash \
      coreutils \
      procps \
      net-tools \
      iproute2 \
      curl \
      wget \
      ca-certificates \
      sudo \
      python3 \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
DOCKERFILE

if [ $? -eq 0 ]; then
  echo "  [OK] ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624 built locally"
else
  echo "  [FAIL] Build failed"
  exit 1
fi
echo ""

# ── 2. ghcr.io/laude-institute/t-bench/python-3-13:20250620 ─────────
# Python 3.13 base (used by ~8 tasks)
echo "[2/2] Building t-bench/python-3-13 equivalent..."
docker build -t ghcr.io/laude-institute/t-bench/python-3-13:20250620 -f - . <<'DOCKERFILE'
FROM python:3.13-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      tmux \
      asciinema \
      bash \
      coreutils \
      procps \
      net-tools \
      iproute2 \
      curl \
      wget \
      ca-certificates \
      sudo \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
DOCKERFILE

if [ $? -eq 0 ]; then
  echo "  [OK] ghcr.io/laude-institute/t-bench/python-3-13:20250620 built locally"
else
  echo "  [FAIL] Build failed"
  exit 1
fi
echo ""

# ── 3. Verify ───────────────────────────────────────────────────────
echo "=== Verification ==="
docker images | grep -E "t-bench|ubuntu" | head -10
echo ""

# Quick test: build task 996 using the local base
echo "=== Test: build task 996 ==="
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL
timeout 120 docker compose -p build_run \
  -f terminal-rl/dataset/seta_env/996/docker-compose.yaml build 2>&1 | tail -5
if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo ""
  echo "=== SUCCESS ==="
  echo "All task Dockerfiles will now use the locally-built base image."
  echo "No network access to ghcr.io needed."
  # Cleanup test
  docker compose -p build_run -f terminal-rl/dataset/seta_env/996/docker-compose.yaml down 2>/dev/null || true
else
  echo ""
  echo "=== FAILED ==="
  echo "Check the build output above for errors."
fi
