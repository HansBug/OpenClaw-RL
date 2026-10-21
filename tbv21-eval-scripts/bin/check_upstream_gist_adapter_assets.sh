#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TBV21_HOME="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ADAPTER_DIR="${OPENCLAW_UPSTREAM_GIST_ADAPTER_DIR:-${TBV21_HOME}/runtime/openclaw_camel_adapter_upstream_gist}"
VENDOR_DIR="${OPENCLAW_UPSTREAM_GIST_VENDOR_PYTHONPATH_DIR:-${TBV21_HOME}/runtime/pythonpath_vendor_min}"
OPENCLAW_DIR="${REAL_OPENCLAW_TERMINAL_RL_DIR:-/mnt/shared-storage-user/narmodel/zhangshaoang/openclaw_rl_puyuan_3573bc3e_git/terminal-rl}"
OPENCLAW_REPO="$(cd -- "${OPENCLAW_DIR}/.." && pwd)"
POOL1_SITE_PACKAGES="${OPENCLAW_POOL1_SITE_PACKAGES:-/mnt/shared-storage-user/tangjia/miniconda3/envs/openclaw-terminal-pool1/lib/python3.12/site-packages}"
HARBOR_PYTHON="${TBV21_HARBOR_PYTHON:-/mnt/shared-storage-user/tangjia/miniconda3/envs/terminal-bench/bin/python}"

echo "[INFO] TBv2.1 home: ${TBV21_HOME}"
echo "[INFO] adapter dir: ${ADAPTER_DIR}"
echo "[INFO] vendor dir: ${VENDOR_DIR}"
echo "[INFO] OpenClaw terminal-rl: ${OPENCLAW_DIR}"

for path in \
  "${ADAPTER_DIR}/openclaw_camel_adapter.py" \
  "${ADAPTER_DIR}/test_loader.py" \
  "${VENDOR_DIR}/sglang/srt/function_call/function_call_parser.py" \
  "${VENDOR_DIR}/sglang/srt/entrypoints/openai/protocol.py" \
  "${OPENCLAW_DIR}/inference_client.py" \
  "${OPENCLAW_DIR}/agent_runner.py"; do
  if [ ! -f "${path}" ]; then
    echo "[ERROR] missing required file: ${path}" >&2
    exit 1
  fi
done

echo "[INFO] checking pinned hashes"
(
  cd "${TBV21_HOME}"
  sha256sum -c <<EOF
70f7976802f1490b8b6ee06fed310c7531d38b7e0bfe70641494fc0ffdefc2da  runtime/openclaw_camel_adapter_upstream_gist/openclaw_camel_adapter.py
cb1fddffe932fb1b2420836dbbe263e9a3824af4a83549b25b3991a4487c8291  runtime/openclaw_camel_adapter_upstream_gist/test_loader.py
8af223ebfd197726f38c978d5aa8404e3a3d5004ed3b3fd1d3044b82ae75ea17  runtime/pythonpath_vendor_min/sglang/srt/function_call/function_call_parser.py
2e367922b6d1284eca4ac997a8fd120f4056a24ca6bcfd4b9a0494a7aeda27fc  runtime/pythonpath_vendor_min/sglang/srt/entrypoints/openai/protocol.py
fb67bf76d87753fb64834c16a0d0462c5f2e64496a01a8031e78431b6ef90db6  bin/run_upstream_gist_adapter_subset_qwen3_8b_base_tbv21.sh
EOF
)

sha256sum -c <<EOF
38b68845e15b95a3691593163f5ff76b2e5ff9853f55e85f840aeb34040addf0  ${OPENCLAW_DIR}/inference_client.py
a8944c60e0e6405a22453ddfa997457cfbd19d5eb2541149ffafd936c92847ee  ${OPENCLAW_DIR}/agent_runner.py
EOF

if [ -d "${OPENCLAW_REPO}/.git" ]; then
  head_commit="$(git -C "${OPENCLAW_REPO}" rev-parse HEAD)"
  if [ "${head_commit}" != "3573bc3e9a3fb73acdbb53bf83a6f72ca1ba7aea" ]; then
    echo "[ERROR] unexpected OpenClaw commit: ${head_commit}" >&2
    exit 1
  fi
  echo "[OK] OpenClaw commit ${head_commit}"
else
  echo "[WARN] OpenClaw repo has no .git: ${OPENCLAW_REPO}"
fi

echo "[INFO] checking adapter import and qwen25 parser shim"
PYTHONPATH="${VENDOR_DIR}:${ADAPTER_DIR}:${OPENCLAW_DIR}:${POOL1_SITE_PACKAGES}:${PYTHONPATH:-}" \
  "${HARBOR_PYTHON}" - <<'PY'
from openclaw_camel_adapter import OpenClawCamelAgent
from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

assert OpenClawCamelAgent.name() == "openclaw-camel-agent"
parser = FunctionCallParser(
    [Tool(function=Function(name="shell_exec"))],
    "qwen25",
)
normal, calls = parser.parse_non_stream(
    '<think>ok</think>\n<tool_call>\n{"name":"shell_exec","parameters":{"command":"pwd"}}\n</tool_call>'
)
assert len(calls) == 1 and calls[0].name == "shell_exec"
print("[OK] adapter import and parser shim")
PY

echo "[OK] upstream gist adapter assets are ready"
