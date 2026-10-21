#!/bin/bash
# 在 pjlab worker (受限 K8s pod) 上从零安装 rootless docker + 用户态网络。
# 适用：Ubuntu 22.04 worker pod，普通用户，NOPASSWD sudo，有出网代理。
# 用法：bash /nfs/docker-rootless/setup.sh
# 详细说明见 /nfs/worker_docker_rootless_guide.md
set -euo pipefail

USER_NAME="$(id -un)"
USER_UID="$(id -u)"
USER_HOME="$HOME"

log() { echo "[setup] $*"; }
err() { echo "[setup][error] $*" >&2; exit 1; }

# ---------- 1. 前置检查 ----------
log "running as: $USER_NAME (uid=$USER_UID)"
[ "$USER_UID" = 0 ] && err "请以普通用户身份运行，不要 sudo 整个脚本。"
sudo -n true 2>/dev/null || err "需要 NOPASSWD sudo。"
. /etc/os-release 2>/dev/null || true
log "host OS: ${PRETTY_NAME:-unknown}"
command -v docker >/dev/null || err "请先在 pod 内确保 docker CLI（包 docker.io）已安装。"
command -v dockerd >/dev/null || err "找不到 dockerd 二进制（包 docker.io 应附带）。"

# ---------- 2. APT 依赖 ----------
log "安装依赖：fuse-overlayfs uidmap curl build-essential"
sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    fuse-overlayfs uidmap curl ca-certificates

# ---------- 3. 把当前用户加入 docker 组（仅为可能的兼容情况） ----------
if ! getent group docker >/dev/null; then
    sudo -n groupadd docker
fi
sudo -n usermod -aG docker "$USER_NAME"

# ---------- 4. 下载 docker-rootless-extras + slirp4netns ----------
mkdir -p "$USER_HOME/.local/bin"
cd /tmp
DOCKER_VER="$(docker --version | awk '{print $3}' | tr -d ',')"
log "本地 docker 版本：$DOCKER_VER。拉取对应的 rootless-extras。"
ROOTLESS_URL="https://download.docker.com/linux/static/stable/x86_64/docker-rootless-extras-${DOCKER_VER}.tgz"
curl -fsSL -o rootless-extras.tgz "$ROOTLESS_URL" \
    || err "无法下载 $ROOTLESS_URL。请检查 http_proxy 是否生效或换 docker 版本号。"
rm -rf rext && mkdir rext
tar -xzf rootless-extras.tgz -C rext --strip-components=1
install -m 755 rext/dockerd-rootless.sh \
                rext/dockerd-rootless-setuptool.sh \
                rext/rootlesskit \
                rext/rootlesskit-docker-proxy \
                rext/vpnkit \
                "$USER_HOME/.local/bin/"

log "安装 slirp4netns（apt 的 0.4.3 太旧，直接拉 upstream v1.3.2 binary）"
curl -fsSL -o "$USER_HOME/.local/bin/slirp4netns" \
    https://github.com/rootless-containers/slirp4netns/releases/download/v1.3.2/slirp4netns-x86_64
chmod +x "$USER_HOME/.local/bin/slirp4netns"

# ---------- 5. 修补 dockerd-rootless.sh ----------
# K8s pod 的根 fs 挂载是 master, rootlesskit 默认 propagation=rslave 会被内核拒绝。
log "修补 dockerd-rootless.sh: propagation=rslave -> private"
sed -i 's|--propagation=rslave|--propagation=private|' "$USER_HOME/.local/bin/dockerd-rootless.sh"

# ---------- 6. 配置 rootless daemon ----------
mkdir -p "$USER_HOME/.config/docker"
cat > "$USER_HOME/.config/docker/daemon.json" <<'EOF'
{
  "storage-driver": "fuse-overlayfs"
}
EOF

# ---------- 7. CLI 默认代理（让 docker run/build 自动注入到容器） ----------
# 从当前 shell 的 http_proxy 取（pjlab 通常已经在 ~/enable_proxy 设好）。
PROXY="${http_proxy:-}"
if [ -n "$PROXY" ]; then
    mkdir -p "$USER_HOME/.docker"
    cat > "$USER_HOME/.docker/config.json" <<EOF
{
  "proxies": {
    "default": {
      "httpProxy": "${PROXY}",
      "httpsProxy": "${PROXY}",
      "noProxy": "10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,127.0.0.1,localhost"
    }
  }
}
EOF
    log "已写 ~/.docker/config.json，默认 proxy=$PROXY"
else
    log "未检测到 http_proxy，跳过 ~/.docker/config.json。容器内将没有 proxy。"
fi

# ---------- 8. 启动包装脚本 ----------
log "生成 ~/.local/bin/dockerd-rootless-launch.sh"
cat > "$USER_HOME/.local/bin/dockerd-rootless-launch.sh" <<LAUNCHER
#!/bin/bash
# 在私有 mount namespace 内启动 rootless dockerd。
#  - detach 掉 K8s pod 对 /proc/* /sys/* 的 lxcfs/tmpfs/ro 覆盖（保留 cgroup mount）
#  - 用 devtmpfs 暴露 /dev/net/tun（pod 默认不带，slirp4netns 需要）
exec sudo -n -E unshare -m -- bash -c '
  while :; do
    covers=\$(awk "
      \\\$3==\"fuse.lxcfs\" {print \\\$2; next}
      (\\\$3==\"tmpfs\" || \\\$3==\"proc\" || \\\$3==\"sysfs\") && (\\\$2 ~ /^\/proc\// || \\\$2 ~ /^\/sys\//) && \\\$2 !~ /^\/sys\/fs\/cgroup/ {print \\\$2}
    " /proc/self/mounts | sort -r)
    [ -z "\$covers" ] && break
    progressed=0
    for m in \$covers; do umount -l "\$m" 2>/dev/null && progressed=1; done
    [ "\$progressed" = 0 ] && break
  done
  mkdir -p /run/host-devtmpfs
  mount -t devtmpfs -o nosuid none /run/host-devtmpfs 2>/dev/null
  if [ -e /run/host-devtmpfs/net/tun ]; then
    mkdir -p /dev/net
    mount --bind /run/host-devtmpfs/net/tun /dev/net/tun 2>/dev/null \\
      || { touch /dev/net/tun; mount --bind /run/host-devtmpfs/net/tun /dev/net/tun; }
  fi
  exec runuser -u $USER_NAME -- env \\
    HOME=$USER_HOME \\
    PATH=$USER_HOME/.local/bin:/nfs/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \\
    XDG_RUNTIME_DIR=/run/user/$USER_UID \\
    DOCKERD_ROOTLESS_ROOTLESSKIT_NET=slirp4netns \\
    DOCKERD_ROOTLESS_ROOTLESSKIT_PORT_DRIVER=builtin \\
    $USER_HOME/.local/bin/dockerd-rootless.sh
'
LAUNCHER
chmod +x "$USER_HOME/.local/bin/dockerd-rootless-launch.sh"

# ---------- 9. Shell 集成 ----------
log "生成 ~/.local/bin/docker-env.sh"
cat > "$USER_HOME/.local/bin/docker-env.sh" <<'ENVSH'
# 由 ~/.bashrc source。设置 DOCKER_HOST 让 docker CLI 连到用户态 rootless daemon。
# 进入交互式 shell 时自动后台拉起 daemon（如未运行）。

export PATH="$HOME/.local/bin:$PATH"
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DOCKER_HOST="unix://$XDG_RUNTIME_DIR/docker.sock"
# dockerd 26.1.3 在我们这种 storage=fuse-overlayfs + 自定义 daemon.json 下，
# /v1.45/version 会 nil pointer panic。固定 API 版本绕开 CLI 的协商请求。
export DOCKER_API_VERSION=1.45

docker-up() {
    if [ -S "$XDG_RUNTIME_DIR/docker.sock" ] && \
       curl -fsS --max-time 2 --unix-socket "$XDG_RUNTIME_DIR/docker.sock" \
            http://localhost/_ping >/dev/null 2>&1; then
        return 0
    fi
    mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
    rm -f "$XDG_RUNTIME_DIR/docker.sock"
    nohup "$HOME/.local/bin/dockerd-rootless-launch.sh" \
        > "$HOME/dockerd-rootless.log" 2>&1 < /dev/null &
    disown 2>/dev/null
    for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        sleep 1
        [ -S "$XDG_RUNTIME_DIR/docker.sock" ] && \
            curl -fsS --max-time 2 --unix-socket "$XDG_RUNTIME_DIR/docker.sock" \
                 http://localhost/_ping >/dev/null 2>&1 && return 0
    done
    echo "docker-up: 启动失败，请看 ~/dockerd-rootless.log" >&2
    return 1
}

docker-down() {
    sudo -n pkill -9 -f 'dockerd-rootless' 2>/dev/null
    sudo -n pkill -9 -f 'rootlesskit' 2>/dev/null
    sudo -n pkill -9 -f 'unshare -m' 2>/dev/null
    sudo -n pkill -9 dockerd 2>/dev/null
    sudo -n rm -f "$XDG_RUNTIME_DIR/docker.sock"
    sudo -n rm -rf "$XDG_RUNTIME_DIR/docker" "$XDG_RUNTIME_DIR/dockerd-rootless"
}

# 交互式 shell 自动后台拉起。
case $- in
    *i*) (docker-up >/dev/null 2>&1 &) ;;
esac
ENVSH

# 把 source 这一行追加到 .bashrc。幂等条件用 docker-env.sh 字面量判定，
# 这样无论是 setup.sh 自己写的 mark 块、还是用户手写过的 source 行，都不会重复。
BASHRC="$USER_HOME/.bashrc"
if ! grep -qF 'docker-env.sh' "$BASHRC" 2>/dev/null; then
    {
        echo
        echo '# >>> rootless-docker (worker setup) >>>'
        echo 'if [ -f "$HOME/.local/bin/docker-env.sh" ]; then'
        echo '    source "$HOME/.local/bin/docker-env.sh"'
        echo 'fi'
        echo '# <<< rootless-docker (worker setup) <<<'
    } >> "$BASHRC"
    log "已追加 source 行到 ~/.bashrc"
else
    log "~/.bashrc 已 source docker-env.sh，跳过追加。"
fi

# ---------- 10. 启动并 smoke test ----------
log "启动 rootless daemon..."
# shellcheck disable=SC1091
source "$USER_HOME/.local/bin/docker-env.sh"
docker-down 2>/dev/null || true
docker-up || err "daemon 启动失败，看 ~/dockerd-rootless.log"

log "运行 hello-world..."
docker run --rm hello-world > /tmp/hw.log 2>&1 && grep -q "Hello from Docker" /tmp/hw.log \
    && log "  hello-world OK" || err "hello-world 失败：$(cat /tmp/hw.log | tail -3)"

log "测试容器联网（HTTP 经 proxy）..."
docker run --rm alpine wget --timeout=10 -qO- http://example.com > /tmp/net.log 2>&1 \
    && grep -q "Example Domain" /tmp/net.log \
    && log "  容器联网 OK" || log "  容器联网失败：$(head -3 /tmp/net.log)"

log "完成。新开 bash 即可直接用 docker。"
log "  - docker version 会报 EOF（dockerd 已知 bug），用 docker info 看版本。"
log "  - daemon 日志：~/dockerd-rootless.log"
log "  - 停止：docker-down ；重启：docker-down && docker-up"
