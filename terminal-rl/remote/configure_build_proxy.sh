#!/usr/bin/env bash
# Configure docker to automatically inject proxy into ALL builds.
#
# This uses Docker's native ~/.docker/config.json "proxies" feature:
# https://docs.docker.com/engine/cli/proxy/#configure-the-docker-client
#
# Effect: every `docker build` / `docker compose build` will automatically
# have HTTP_PROXY/HTTPS_PROXY/NO_PROXY set as build-time env vars, without
# modifying any Dockerfile.
#
# Usage (on CPU worker):
#   bash terminal-rl/remote/configure_build_proxy.sh
#
# To undo:
#   rm ~/.docker/config.json  (or remove the "proxies" key)
set -euo pipefail

PROXY_URL="${PROXY_URL:-http://httpproxy-headless.kubebrain.svc.pjlab.local:3128}"
NO_PROXY_LIST="localhost,127.0.0.1,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn"

DOCKER_CONFIG_DIR="${HOME}/.docker"
DOCKER_CONFIG_FILE="${DOCKER_CONFIG_DIR}/config.json"

mkdir -p "$DOCKER_CONFIG_DIR"

# If config.json exists, merge proxies into it; otherwise create fresh
if [ -f "$DOCKER_CONFIG_FILE" ]; then
  # Use python to merge (preserves existing keys like cli-plugins)
  python3 -c "
import json, sys
with open('$DOCKER_CONFIG_FILE') as f:
    cfg = json.load(f)
cfg['proxies'] = {
    'default': {
        'httpProxy': '$PROXY_URL',
        'httpsProxy': '$PROXY_URL',
        'noProxy': '$NO_PROXY_LIST'
    }
}
with open('$DOCKER_CONFIG_FILE', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Updated existing config.json')
"
else
  cat > "$DOCKER_CONFIG_FILE" <<EOF
{
  "proxies": {
    "default": {
      "httpProxy": "$PROXY_URL",
      "httpsProxy": "$PROXY_URL",
      "noProxy": "$NO_PROXY_LIST"
    }
  }
}
EOF
  echo "Created new config.json"
fi

echo ""
echo "=== Current ~/.docker/config.json ==="
cat "$DOCKER_CONFIG_FILE"
echo ""
echo ""
echo "=== Verification ==="
echo "All future 'docker build' / 'docker compose build' will automatically"
echo "inject these env vars into build containers:"
echo "  HTTP_PROXY=$PROXY_URL"
echo "  HTTPS_PROXY=$PROXY_URL"
echo "  NO_PROXY=$NO_PROXY_LIST"
echo ""
echo "No Dockerfile modifications needed."
echo ""
echo "=== Quick test ==="
echo "Running: docker build with a test that checks proxy is injected..."
docker build --no-cache -t proxy-test -f - . <<'DOCKERFILE' 2>&1 | tail -5
FROM ubuntu:24.04
RUN echo "http_proxy=$http_proxy" && echo "HTTP_PROXY=$HTTP_PROXY" && apt-get update && echo "APT UPDATE OK"
DOCKERFILE

if [ $? -eq 0 ]; then
  echo "[OK] Proxy injection working. apt-get update succeeded inside build."
  docker rmi proxy-test >/dev/null 2>&1 || true
else
  echo "[FAIL] Something went wrong. Check output above."
fi
