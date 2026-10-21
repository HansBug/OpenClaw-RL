"""Local Harbor runtime patches for this TB v2.1 bundle.

This file is loaded only when TBV21_HOME/patches is placed on PYTHONPATH.
It keeps shared conda packages untouched while making Terminus-2 setup
reliable in proxied Docker containers.
"""

from __future__ import annotations

import os
from pathlib import Path


def _patch_terminus2_apt_proxy() -> None:
    try:
        from harbor.agents.terminus_2.tmux_session import TmuxSession
    except Exception:
        return

    original = TmuxSession._get_combined_install_command

    def patched(self, system_info, tools):  # type: ignore[no-untyped-def]
        package_manager = system_info.get("package_manager")
        if package_manager != "apt-get":
            return original(self, system_info, tools)

        packages = " ".join(tools)
        proxy = (
            os.environ.get("HTTP_PROXY")
            or os.environ.get("http_proxy")
            or "http://100.103.187.133:18080"
        )
        https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or proxy

        apt_conf = (
            "printf '%s\\n' "
            f"'Acquire::http::Proxy \"{proxy}\";' "
            f"'Acquire::https::Proxy \"{https_proxy}\";' "
            "'Acquire::http::Timeout \"15\";' "
            "'Acquire::https::Timeout \"15\";' "
            "'Acquire::Retries \"2\";' "
            "> /etc/apt/apt.conf.d/95tbv21-proxy"
        )
        apt_opts = (
            "-o Acquire::http::Timeout=15 "
            "-o Acquire::https::Timeout=15 "
            "-o Acquire::Retries=2 "
            f"-o Acquire::http::Proxy={proxy} "
            f"-o Acquire::https::Proxy={https_proxy}"
        )
        return (
            f"{apt_conf} && "
            f"DEBIAN_FRONTEND=noninteractive apt-get {apt_opts} update && "
            f"DEBIAN_FRONTEND=noninteractive apt-get {apt_opts} install -y {packages}"
        )

    TmuxSession._get_combined_install_command = patched


def _patch_terminus2_tmux_bundle() -> None:
    try:
        from harbor.agents.terminus_2.terminus_2 import Terminus2Agent
        from harbor.agents.terminus_2.tmux_session import TmuxSession
    except Exception:
        return

    original_init = Terminus2Agent.__init__
    original_attempt = TmuxSession._attempt_tmux_installation

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if os.environ.get("TBV21_DISABLE_ASCIINEMA", "1") == "1":
            kwargs["record_terminal_session"] = False
        return original_init(self, *args, **kwargs)

    async def patched_attempt(self):  # type: ignore[no-untyped-def]
        bundle = Path(
            os.environ.get(
                "TBV21_TMUX_BUNDLE",
                "/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1/runtime/tmux-runtime-linux-x86_64.tar.gz",
            )
        )
        if bundle.is_file():
            await self.environment.upload_file(bundle, "/tmp/tbv21-tmux-runtime.tar.gz")
            install_cmd = (
                "set -e; "
                "mkdir -p /opt/tbv21-tmux /usr/local/bin; "
                "tar -xzf /tmp/tbv21-tmux-runtime.tar.gz -C /opt/tbv21-tmux; "
                "cat > /usr/local/bin/tmux <<'EOF'\n"
                "#!/bin/sh\n"
                "export LD_LIBRARY_PATH=/opt/tbv21-tmux/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}\n"
                "exec /opt/tbv21-tmux/bin/tmux \"$@\"\n"
                "EOF\n"
                "chmod +x /usr/local/bin/tmux; "
                "/usr/local/bin/tmux -V"
            )
            result = await self.environment.exec(command=install_cmd, user="root")
            if result.return_code == 0:
                self._logger.debug("Installed tmux from TBV21 bundle")
                return
            self._logger.warning(
                "TBV21 tmux bundle install failed: %s",
                (result.stderr or result.stdout or "").strip(),
            )

        return await original_attempt(self)

    Terminus2Agent.__init__ = patched_init
    TmuxSession._attempt_tmux_installation = patched_attempt


_patch_terminus2_apt_proxy()
_patch_terminus2_tmux_bundle()
