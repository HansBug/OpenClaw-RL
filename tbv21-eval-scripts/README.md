# TBv2.1 Harbor Full Eval 无人值守运行手册

集群 README 真实路径：`/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1/README.md`。如果要在类似当前的 PJLab GPU worker 环境中配置并运行 TBv2.1 Harbor full eval，直接把这个 README 路径交给 Codex 或 Claude Code，让其按文档执行即可。

本文档记录 2026-07-01 至 2026-07-02 在 GPU worker 上跑通的一整套流程：使用本机 rootless Docker 承载 TBv2.1 task containers，使用本机 SGLang 提供 Qwen3-8B OpenAI-compatible endpoint，然后通过 Harbor 完整运行 Terminal-Bench v2.1 eval。

目标不是“某次人工救火跑完”，而是把 Harbor、Docker、代理、SGLang、镜像预拉、配置自检、监控和结果汇总串成一个 agent 可以照做的 runbook。

## 2026-07-03 Qwen3-8B-base modeB aligned 入口

### 上游 gist adapter + puyuan 3573bc3e 复测入口

如果目标是按 #23/#25 最新 comment 指向的上游 adapter 复测 #25 里 TB2 通过的两个 task，先读：

```text
docs/upstream_gist_adapter_tbv21_20260703.md
docs/upstream_gist_adapter_maintenance_20260703.md
```

这一轮使用 `runtime/openclaw_camel_adapter_upstream_gist/openclaw_camel_adapter.py`，SHA256 `70f7976802f1...`，并把 OpenClaw-RL `dev-agenticrl-safety-exploration-harness` / `3573bc3e9a3fb73acdbb53bf83a6f72ca1ba7aea` 持久化到：

```text
/mnt/shared-storage-user/narmodel/zhangshaoang/openclaw_rl_puyuan_3573bc3e_git
```

复跑入口：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
bash bin/check_upstream_gist_adapter_assets.sh
TBV21_UPSTREAM_GIST_ATTEMPTS=3 \
TBV21_UPSTREAM_GIST_CONCURRENCY=1 \
TBV21_PREPULL_IMAGES=1 \
bin/run_upstream_gist_adapter_subset_qwen3_8b_base_tbv21.sh hf-model-inference modernize-scientific-stack
```

当前结果：6/6 trial 无 Harbor exception，`modernize-scientific-stack` 为 2/3 PASS，`hf-model-inference` 为 0/3 PASS。对 #25 的两个 solved task 只能说部分复现：`modernize` 对上了方向，`hf` 未复现。

### 本地复刻 adapter 历史记录

如果目标是复查 `qwen3-8b-base` 在 TBv2.1 modeB aligned 小测中的配置/协议对齐问题，先读：

```text
docs/qwen3_base_protocolfix_20260703.md
```

相关持久化改动：

| 文件 | 用途 |
|---|---|
| `runtime/openclaw_camel_adapter/openclaw_camel_adapter.py` | 当前 adapter：`modeB-qwen3-chat-template-terminaltoolkit-20260703-inputids6-issue25-visible-sampling`；复刻 #25 modeB 的 Qwen3 `chat_template`、SGLang `/generate` `input_ids`、qwen25 XML 工具格式和 TerminalToolkit schema；默认 sampling payload 按 #25 原始轨迹可见字段执行 |
| `bin/run_protocolfix_subset_qwen3_8b_base_tbv21.sh` | 只跑 #25 曾 pass 的 `hf-model-inference`、`modernize-scientific-stack` 小测，不跑全量 |
| `docs/qwen3_base_protocolfix_20260703.md` | 记录配置逐项对齐、prompt token 对齐证据、inputids1/3/4/5/6 小测、残余差异和后续建议 |
| `/mnt/shared-storage-user/narmodel/zhangshaoang/openclaw_rl_hansbug` | OpenClaw-RL 上游源码持久化副本，commit `f5107bcea45b772f2f62d4f46d5c37be0999dd10` |
| `state/upstream/issue25_reference/` | #25 PASS 原始轨迹参考文件的持久化副本，不再依赖 `/tmp` |

最小复现命令：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
TBV21_PROTOCOLFIX_ATTEMPTS=1 \
TBV21_PROTOCOLFIX_CONCURRENCY=1 \
TBV21_PREPULL_IMAGES=0 \
bin/run_protocolfix_subset_qwen3_8b_base_tbv21.sh
```

当前结论：最早的 parser-only `protocolfix2` 已废弃；当前路线是按 #25 modeB 重建协议栈。已确认 `modernize-scientific-stack` 在 TBv2.1 小测中 `3/3 PASS`，且 prompt token 与 #25 PASS 轨迹精确对齐；`hf-model-inference` 的首轮 prompt 已与 #25 参考轨迹 token 级完全一致，工具 schema 也一致，但在 `inputids3/inputids4/inputids5/inputids6` 小测中仍为 `0/3 PASS`。`inputids6` 是当前默认 #25-visible sampling payload（不传 `sampling_seed/no_stop_trim/spaces_between_special_tokens`），失败已落到首轮采样轨迹分叉后的 Flask 应用代码质量，而不是 Harbor/Docker/apt/uv/SGLang/tool execution 基础设施崩溃。当前不建议跑全量，应继续围绕 `hf-model-inference` 缩小 SGLang 版本、TP、上游 adapter bit-identical 程度以及 TB2→TBv2.1 task/verifier 差异。

持久化边界：源码、adapter、runbook、job 结果、配置、#25 参考轨迹都在 `/mnt/shared-storage-user/narmodel/zhangshaoang/` 下；`/tmp` 只保留为 rootless Docker layer、uv cache、容器内 session log 这类可重建 scratch。清空 `/tmp` 后应能按本文档从持久化目录重启基础设施和小测。

已验证结果：

- 工作目录：`/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1`
- Docker：rootless Docker，`DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock`；本次用户 `zhangshaoang` 的实际 UID 为 `10250`
- Docker storage driver：`fuse-overlayfs`
- Docker root：`/tmp/tbv21-rootless-docker-zhangshaoang`
- Model：`qwen3-8b-rl-iter215`
- SGLang：`http://127.0.0.1:30000/v1`
- Harbor job：`full_eval_tbv21_qwen3-8b-rl-iter215_ready_20260701_215702`
- Eval 时间：`2026-07-01 22:00:13` 到 `2026-07-02 07:22:00`
- 完成度：`89/89`
- Harbor 聚合分数：`2.0 / 89 = 0.0224719101`
- 任务级 timeout：`AgentTimeoutError: 20`
- reward=1 的任务：`configure-git-webserver__zVAaSVv`、`hf-model-inference__72rnD2H`
- 收尾状态：rootless Docker 无残留 task container；SGLang endpoint 健康，保留运行便于复用。

## 0. 一句话结论

在该 worker 上，推荐路线是：

1. 启动 rootless Docker，并固定到 `/run/user/$(id -u)/docker.sock`。
2. 在 Docker daemon 和 Harbor task container 两层都注入代理。
3. eval 前预拉全部 TBv2.1 Docker images。
4. 启动或复用 SGLang 的 OpenAI-compatible endpoint。
5. 生成 Harbor config，并强制自检 `proxy env`、`DEBIAN_FRONTEND=noninteractive`、`TZ=Etc/UTC`、`model_info`。
6. 在 tmux 中启动完整 Harbor eval；当前 worker 使用 `gpu:4.0`，新 worker 若没有该 pane，先创建或改用 `tbv21_full_eval:0.0`。
7. 用监控脚本轮询 job result、Docker container、SGLang health，直到 `finished_at != null`。

不要在 task 运行中手工修改单个 task。除非 Docker/SGLang/Harbor 主流程真的坏了，否则 task timeout 应视为 eval 结果，不是 infra 故障。

## 1. 当前集群里的配置全景

这一节是迁移时最重要的部分。当前 worker 不是普通裸机，而是受限的 GPU worker / K8s pod 类环境：有 GPU、有共享存储、有代理、有部分 sudo 能力，但不能假设宿主 Docker socket、systemd Docker daemon 或 privileged Docker-in-Docker 都可用。

### 1.1 分层拓扑

```text
GPU worker pod
  |
  |-- /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
  |     |-- scripts / dataset / runtime assets / logs / jobs / state
  |
  |-- SGLang model server
  |     |-- binds GPU 0,1,2,3
  |     |-- listens on 127.0.0.1:30000
  |     |-- serves OpenAI-compatible /v1 endpoint
  |
  |-- rootless Docker daemon
  |     |-- listens on unix:///run/user/$(id -u)/docker.sock
  |     |-- data-root=/tmp/tbv21-rootless-docker-$USER
  |     |-- storage-driver=fuse-overlayfs
  |     |-- network=slirp4netns via rootlesskit
  |
  |-- Harbor
        |-- launched from bin/harbor wrapper
        |-- talks to SGLang through http://127.0.0.1:30000/v1
        |-- talks to Docker through DOCKER_HOST after bin/use_worker_rootless_docker.sh start
        |-- starts TBv2.1 task containers
        |-- injects proxy + noninteractive apt + tmux/uv/wheelhouse runtime mounts
```

### 1.2 为什么必须这样配

| 层 | 当前集群约束 | 必须配置 | 不这样做的失败模式 |
|---|---|---|---|
| Docker daemon | pod 内普通 nested dockerd 会遇到 cgroup/capability 限制 | rootless Docker + `fuse-overlayfs` + `slirp4netns` | `mkdir /sys/fs/cgroup/... read-only file system`、`docker run` 失败 |
| Docker data root | GPFS 对 layer unpack 的 owner/chown 操作不友好 | rootless data root 放 `/tmp/tbv21-rootless-docker-$USER` | unpack image layer 时报 `Lchown ... operation not permitted` |
| Docker API | dockerd 26.1.3 在该组合下 `/version` 协商可能 EOF/panic；`env.sh` 只是默认写 rootless socket 路径，真正可用要先启动 daemon | `DOCKER_API_VERSION=1.45`；先 `bash bin/use_worker_rootless_docker.sh start`，再 `source "$TBV21_DOCKER_HOST_ENV"`，用 `docker info` 判断健康 | `docker version` 报 EOF被误判；或 `source env.sh` 后直接 `docker info` 连不上未启动的 socket |
| Docker pull | Docker Hub 直连可能 TLS timeout | daemon registry mirror + pre-pull + `docker.1ms.run` fallback | full eval 中途卡镜像拉取或失败 |
| 容器网络 | task 容器内访问 apt/pypi/外部数据需要代理 | Docker CLI proxy + Harbor env proxy + APT_CONFIG | apt/pip/download 超时 |
| apt 交互 | `tzdata` 等包可能等待交互输入 | `DEBIAN_FRONTEND=noninteractive` + `TZ=Etc/UTC` | task 卡住，看似 Docker/Harbor 停滞 |
| Harbor 模型配置 | LiteLLM 不一定认识本地 served model | `model_info.max_input_tokens/max_output_tokens` | unknown model/context fallback |
| SGLang 管理 | pid 文件可能缺失但服务仍健康 | 先查 `/v1/models`，健康则复用 | 误杀正在工作的模型服务 |
| tmux/runtime | task container 里需要 tmux，镜像内未必合适 | bind mount `runtime/tmux_runtime` 到 `/opt/tbv21-tmux` | agent terminal session 初始化失败 |
| Python tools | task 里需要 `uv/curl/apt-get` 可控路径 | bind mount `runtime/tools` 和 `runtime/wheelhouse` | pip/uv 网络不稳定，依赖安装失败 |

### 1.3 当前方案和其他方案的边界

当前跑通的是“本 GPU worker 内 rootless Docker”路线。它和 CPU worker docker-env-server 不是同一个东西：

- CPU worker docker-env-server：GPU worker 通过 HTTP 访问远端 pool_server，用于 OpenClaw RL rollout 环境服务。
- 本文路线：Harbor 直接用 Docker backend 在本 GPU worker 内启动 TBv2.1 task containers。

不要把 `WORKER_URLS=http://<cpu>:18081` 当成 Harbor Docker backend。Harbor 需要的是 Docker API / Docker CLI 能 `docker run`、`docker compose`、创建网络和 bind mounts。

理论替代方案：

- 远程 Docker daemon TCP API：如果可信开发机暴露 Docker API，可设置 `DOCKER_HOST=tcp://100.x.x.x:2376`。当前这次没有用。
- privileged rjob/rlaunch 里直接跑 dockerd：如果集群允许 privileged + fuse/nvidia runtime，可行。当前这次没有用。
- 宿主 `/var/run/docker.sock` 挂载：如果 worker 有 host Docker socket，可行。当前这次没有用。

本文后续命令默认采用已验证的 rootless Docker 路线。

## 2. 当前运行状态与网络资源加载方案

### 2.1 当前状态

截至 2026-07-02，本机状态是：

- 上一轮 Harbor full eval 已完成，不是半路卡住。
- `jobs/full_eval_tbv21_qwen3-8b-rl-iter215_ready_20260701_215702/result.json` 存在，且 `finished_at=2026-07-02T07:22:00.608529`。
- `n_total_trials=89`，实际 result files 为 `89`。
- `error_counts={'AgentTimeoutError': 20}`。这些是 task/model 层超时，不是 Docker、SGLang 或 Harbor 主流程崩溃。
- `reward_sum=2.0`，Harbor 聚合口径 `2.0 / 89 = 0.02247191011235955`。
- rootless Docker 可用，`docker info` 显示 `ServerVersion=26.1.3`、`Storage Driver=fuse-overlayfs`、`Docker Root Dir=/tmp/tbv21-rootless-docker-zhangshaoang`、`Cgroup Driver=none`。
- 当前没有残留 Harbor task containers。
- SGLang endpoint 健康，`curl --noproxy '*' http://127.0.0.1:30000/v1/models` 返回 `qwen3-8b-rl-iter215`。

核验命令：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
source ./env.sh
source "$TBV21_DOCKER_HOST_ENV"

docker info --format 'server={{.ServerVersion}} root={{.DockerRootDir}} driver={{.Driver}} cgroup={{.CgroupDriver}} containers={{.Containers}} images={{.Images}}'
docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:30000/v1/models

python3 - <<'PY'
from pathlib import Path
import json, collections
job=Path("jobs/full_eval_tbv21_qwen3-8b-rl-iter215_ready_20260701_215702")
res=json.loads((job/"result.json").read_text())
results=list(job.glob("*/result.json"))
errs=collections.Counter()
reward_sum=0.0
for p in results:
    d=json.loads(p.read_text())
    exc=d.get("exception_info")
    if exc:
        errs[exc.get("exception_type") or "unknown"] += 1
    reward=((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    reward_sum += reward or 0.0
print("started_at=", res.get("started_at"))
print("finished_at=", res.get("finished_at"))
print("n_total_trials=", res.get("n_total_trials"), "result_files=", len(results))
print("error_counts=", dict(errs))
print("reward_sum=", reward_sum, "mean_all=", reward_sum/(res.get("n_total_trials") or len(results)))
PY
```

### 2.2 网络资源加载的四层处理

TBv2.1 的很多 Docker images 或 task 运行过程会访问公网或内网镜像，例如：

- `docker pull alexgshaw/...`
- task container 内 `apt-get update` / `apt-get install`
- task container 内 `pip` / `uv` 安装依赖
- task 自己下载 benchmark 数据、模型文件或测试资源

本方案不是靠手工改某个 task，而是在四层统一处理。

**第一层：Docker image 预拉和 mirror fallback**

完整 eval 前必须先跑：

```bash
source ./env.sh
source "$TBV21_DOCKER_HOST_ENV"
bash bin/prepull_tbv21_images.sh "$TBV21_TASKS_DIR"
```

脚本会从 89 个 `task.toml` 解析 `docker_image`，逐个 `docker pull`。如果 Docker Hub 直连失败，会尝试：

```text
docker.1ms.run/<原始 image>
```

成功后再 `docker tag` 回原始 image 名称。这样 Harbor 真正跑 eval 时不再边执行边等大镜像下载。

本次证据：

```text
state/tbv21_task_images_20260701_215702.tsv
state/tbv21_task_images_failed_20260701_215702.tsv  # 0 bytes
prepull summary: total=89 ok=89 failed=0
```

**第二层：rootless Docker daemon 代理和 registry mirror**

rootless Docker daemon 的实际配置在 `~/.config/docker/daemon.json`：

```json
{
  "storage-driver": "fuse-overlayfs",
  "data-root": "/tmp/tbv21-rootless-docker-zhangshaoang",
  "registry-mirrors": [
    "https://docker.1ms.run"
  ],
  "proxies": {
    "http-proxy": "http://100.103.187.133:18080",
    "https-proxy": "http://100.103.187.133:18080",
    "no-proxy": "10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,127.0.0.1,localhost"
  }
}
```

这解决 daemon 自己拉镜像和访问 registry 时的网络问题。

**第三层：Docker CLI 和 task container 环境变量**

`env.sh` 会按当前 shell 的代理重写 `docker-cli/config.json`。下面是生成后的 JSON 示例，实际内容由当前环境变量自动填充：

```json
{
  "proxies": {
    "default": {
      "httpProxy": "http://100.103.187.133:18080",
      "httpsProxy": "http://100.103.187.133:18080",
      "noProxy": "10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,127.0.0.1,localhost"
    }
  }
}
```

Harbor config 也会给 task container 注入：

```text
HTTP_PROXY
HTTPS_PROXY
NO_PROXY
http_proxy
https_proxy
no_proxy
DEBIAN_FRONTEND=noninteractive
TZ=Etc/UTC
APT_CONFIG=/opt/tbv21-apt/apt.conf
```

这样 Docker `run/build` 和容器内大部分命令都能继承正确代理。

**第四层：APT/uv/pip 的专门配置**

APT 不总是可靠读取普通环境变量，所以脚本显式生成并挂载：

```text
runtime/apt/apt.conf
```

内容：

```text
Acquire::http::Proxy "http://100.103.187.133:18080";
Acquire::https::Proxy "http://100.103.187.133:18080";
Acquire::Retries "3";
Acquire::http::Timeout "30";
Acquire::https::Timeout "30";
```

Python 依赖侧，Harbor task container 里设置：

```text
UV_INDEX_URL=http://mirrors.h.pjlab.org.cn/pypi/simple/
UV_EXTRA_INDEX_URL=http://pypi.i.h.pjlab.org.cn/brain/dev/+simple
UV_TRUSTED_HOST=mirrors.h.pjlab.org.cn pypi.i.h.pjlab.org.cn
UV_FIND_LINKS=/opt/tbv21-wheelhouse
UV_CACHE_DIR=/tmp/tbv21-uv-cache
UV_LINK_MODE=copy
```

`runtime/wheelhouse` 当前主要缓存 pytest 生态：

```text
pytest
pytest_json_ctrf
pluggy
iniconfig
packaging
pygments
typing_extensions
exceptiongroup
tomli
```

注意：wheelhouse 不是全量离线依赖库。其他依赖仍依赖 PJLab PyPI mirror 或外网代理可达。

### 2.3 网络问题的剩余边界

上述四层能解决大多数基础设施网络问题，但仍有一个边界：task 自身可能在运行中访问外部数据源，且该数据源慢、失效或体积很大。这类情况会表现为 task 级 timeout，而不是 Docker/Habor/SGLang 故障。

本次 `caffe-cifar-10` 就是典型例子：加入 `DEBIAN_FRONTEND=noninteractive` 和 `TZ=Etc/UTC` 后不再卡 `tzdata` 交互，但后续仍可能因为 CIFAR 外部下载或任务本身耗时触发 `AgentTimeoutError`。这类结果应该记录为 eval 结果，不建议临场修改 task。

## 3. `tbv2.1` bundle 静态资源盘点

路径：

```bash
export TBV21_HOME=/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
cd "$TBV21_HOME"
```

目录大小是 2026-07-02 的一次盘点，供判断迁移成本：

```text
67M   bin
61M   docker-cli-plugins
127M  datasets
98M   envs
68M   runtime
1.0K  models
1.9M  state
28M   logs
2.4G  jobs
1.6G  cache
238M  tmp
404M  pythons
257K  docs
5.0G  total
```

### 3.1 必须保留的预置资源

| 路径 | 类型 | 用途 | 新环境怎么验证 |
|---|---|---|---|
| `env.sh` | 配置入口 | 统一设置 TBV21 paths、proxy、Docker、SGLang、model env | `source ./env.sh && echo $TBV21_HOME` |
| `bin/harbor` | wrapper | 调 Harbor CLI | `bin/harbor --version` |
| `bin/python_sglang` | wrapper | 调 SGLang Python | `bin/python_sglang -c 'import torch; print(torch.cuda.device_count())'` |
| `bin/setup_worker_rootless_docker.sh` | 安装脚本 | 首次配置 rootless Docker | `test -x bin/setup_worker_rootless_docker.sh` |
| `bin/use_worker_rootless_docker.sh` | 启停脚本 | 启动/停止/检查 rootless Docker | `bash bin/use_worker_rootless_docker.sh status` |
| `bin/prepull_tbv21_images.sh` | 镜像预拉 | 从 `task.toml` 提取并拉取 89 个镜像 | `bash bin/prepull_tbv21_images.sh "$TBV21_TASKS_DIR"` |
| `bin/start_sglang.sh` | 模型服务 | 启动或复用 SGLang | `bash bin/start_sglang.sh 0,1,2,3` |
| `bin/run_full_eval_qwen3_8b.sh` | 主流程 | 生成 config 并运行完整 eval | `bash -n bin/run_full_eval_qwen3_8b.sh` |
| `bin/run_one_task_eval_qwen3_8b.sh` | smoke | 单任务 eval | `bash bin/run_one_task_eval_qwen3_8b.sh regex-chess` |
| `datasets/terminal-bench-2-1/tasks` | dataset | TBv2.1 任务集，当前 89 个 task | `find ... -mindepth 1 -maxdepth 1 -type d | wc -l` |
| `runtime/tmux_runtime` | 静态 runtime | task container 里的 tmux | `test -x runtime/tmux_runtime/bin/tmux` |
| `runtime/tools` | 静态 runtime | task container 里的 `uv`、`uvx`、`curl`、`apt-get` wrapper | `test -x runtime/tools/bin/uv` |
| `runtime/wheelhouse` | 离线 wheel | pytest/ctrf 等常用 wheels，减少容器内下载 | `ls runtime/wheelhouse/*.whl` |
| `docker-cli/config.json` | 半静态 Docker CLI 配置 | 自动给 build/run 注入 proxy；`env.sh` 默认会按当前代理重写 | `source ./env.sh && cat docker-cli/config.json` |
| `docker-cli-plugins/docker-compose` | Compose V2 真身 | Harbor Docker backend 需要 compose | `docker compose version` |
| `models/qwen3-8b-rl-iter215` | symlink | 指向实际模型 checkpoint | `readlink -f models/qwen3-8b-rl-iter215` |

Docker CLI/Compose 相关目录关系：

```text
docker-cli/config.json
  Docker CLI 配置目录；env.sh 设置 DOCKER_CONFIG 指向这里，并写入 proxies。

docker-cli/cli-plugins
  env.sh 自动创建的 symlink，用来让 Docker CLI 找到 compose plugin。

cli-plugins/
  bundle 根目录下的兼容目录。

docker-cli-plugins/docker-compose
  Compose V2 二进制真身。
```

代理规则很重要：`env.sh` 会优先继承当前 shell 已有的 `HTTP_PROXY/HTTPS_PROXY`，否则使用 `TBV21_PROXY_URL`，再否则使用默认 PJLab headless proxy。换 worker 或换代理时，推荐显式设置后重新 source：

```bash
export TBV21_PROXY_URL=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128
export HTTP_PROXY="$TBV21_PROXY_URL"
export HTTPS_PROXY="$TBV21_PROXY_URL"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export NO_PROXY='localhost,127.0.0.1,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,.pjlab.local,.svc'
export no_proxy="$NO_PROXY"
source ./env.sh
cat docker-cli/config.json
```

如果需要保留手写 `docker-cli/config.json`，设置：

```bash
export TBV21_DOCKER_CONFIG_REWRITE=0
source ./env.sh
```

默认建议不要这样做，因为旧 proxy 锁在 `docker-cli/config.json` 里会导致 task container 内 apt/pip/download 卡住。

### 3.2 外部依赖：模型 symlink 和 shared conda

`models/qwen3-8b-rl-iter215` 不是实际模型文件，而是 symlink。脚本名里的 `qwen3_8b` 是历史命名，脚本本体读取 `TBV21_MODEL_NAME` 和 `TBV21_MODEL_PATH`，换模型时不需要改脚本名。

```bash
ls -la models
readlink -f models/qwen3-8b-rl-iter215
```

当前目标：

```text
/mnt/shared-storage-user/puyuan/tangjia/model/models--HansBug--Qwen3-8B-OpenClaw-RL-iter215
```

迁移到新 worker 时必须验证该 symlink 目标可读，不只是目录存在：

```bash
MODEL_REAL="$(readlink -f models/qwen3-8b-rl-iter215)"
test -r "$MODEL_REAL/config.json" || {
  echo "[ERROR] model symlink target missing or unreadable: $MODEL_REAL"
  exit 1
}
```

如果新环境看不到这个路径，可以改成新的 checkpoint：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
ln -sfn /path/to/qwen3-8b-checkpoint models/qwen3-8b-rl-iter215
export TBV21_MODEL_PATH="$PWD/models/qwen3-8b-rl-iter215"
export TBV21_MODEL_NAME=qwen3-8b-rl-iter215
```

`bin/harbor` 和 `bin/python_sglang` 当前是 wrapper，指向 shared conda。这个依赖在文档开头就要确认；如果不可读，不要继续跑 full eval，先修 Python 环境。

```bash
cat bin/harbor
# exec /mnt/shared-storage-user/tangjia/miniconda3/envs/terminal-bench/bin/harbor "$@"

cat bin/python_sglang
# exec /mnt/shared-storage-user/tangjia/miniconda3/envs/slime/bin/python "$@"
```

验证 wrapper 目标：

```bash
grep -E '/mnt/shared-storage-user/.*/bin/' bin/harbor bin/python_sglang
bin/harbor --version
bin/python_sglang - <<'PY'
import importlib.util, torch
print("cuda_count", torch.cuda.device_count())
for name in ["sglang", "openai", "transformers"]:
    print(name, bool(importlib.util.find_spec(name)))
PY
```

迁移到新 worker 时，如果 `/mnt/shared-storage-user/tangjia/miniconda3/...` 不存在，有两种做法：

1. 恢复同路径 shared conda。
2. 按 `docs/rebuild_envs.md` 在 `envs/harbor` 和 `envs/sglang` 下重建本地环境，然后更新 wrappers。

本地 wrapper 模板：

```bash
cat > bin/harbor <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/envs/harbor/bin/harbor" "$@"
EOF
chmod +x bin/harbor

cat > bin/python_sglang <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/envs/sglang/bin/python" "$@"
EOF
chmod +x bin/python_sglang
```

### 3.3 运行产物和可清理资源

这些不是启动新 eval 的必要静态资源：

| 路径 | 内容 | 是否可清理 |
|---|---|---|
| `jobs/` | Harbor eval 输出，包含本次完整结果 | 保留结果时不要清；新环境可只保留 `full_eval...ready...` 作参考 |
| `logs/` | pre-pull、SGLang、full eval 日志 | 可归档 |
| `state/*.config.json` | 历史 Harbor config | 可保留作模板 |
| `state/tbv21_task_images_*.tsv` | 镜像清单 | 可复用，也可重新生成 |
| `cache/uv` | uv 下载缓存 | 可保留加速；不是必需 |
| `tmp/` | 运行临时目录 | 可清理 |
| `docker-data-rootless/` | 旧 rootless Docker data root | 当前推荐不用 GPFS 上这个目录；不要依赖它 |

不要清理：

- `datasets/`
- `runtime/`
- `bin/`
- `docker-cli*`
- `env.sh`
- 可用的 `models` symlink 或实际模型目录

## 4. 新 worker 复用该 bundle 的 checklist

在一个“类似当前”的新 GPU worker 中，按下面顺序做。agent 不应该跳步。

### 4.1 基础路径和资源验证

```bash
set -euo pipefail

export TBV21_HOME=/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
cd "$TBV21_HOME"
source ./env.sh

echo "[INFO] checking bundle resources"
test -d "$TBV21_TASKS_DIR"
test "$(find "$TBV21_TASKS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | awk '{print $1}')" = "89"
test -x bin/harbor
test -x bin/python_sglang
test -x runtime/tmux_runtime/bin/tmux
test -x runtime/tools/bin/uv
test -d runtime/wheelhouse
test -x docker-cli-plugins/docker-compose

MODEL_REAL="$(readlink -f "$TBV21_MODEL_PATH")"
test -r "$MODEL_REAL/config.json" || { echo "[ERROR] missing or unreadable model config: $MODEL_REAL/config.json"; exit 1; }

bin/harbor --version
bin/python_sglang - <<'PY'
import importlib.util
import torch
print("cuda_count", torch.cuda.device_count())
for name in ["sglang", "openai", "transformers"]:
    print(name, bool(importlib.util.find_spec(name)))
PY
```

也可以先跑 bundle 自带 doctor：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
bash bin/doctor.sh
cat state/doctor_latest.txt
```

如果 doctor 和上面的手工检查结论冲突，以手工检查为准，并把缺失项补齐后再继续。

一条命令盘点关键静态资源：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
for p in \
  env.sh \
  bin/harbor \
  bin/python_sglang \
  bin/setup_worker_rootless_docker.sh \
  bin/use_worker_rootless_docker.sh \
  bin/prepull_tbv21_images.sh \
  bin/start_sglang.sh \
  bin/run_full_eval_qwen3_8b.sh \
  datasets/terminal-bench-2-1/tasks \
  runtime/tmux_runtime/bin/tmux \
  runtime/tools/bin/uv \
  runtime/wheelhouse \
  docker-cli/config.json \
  docker-cli-plugins/docker-compose \
  models/qwen3-8b-rl-iter215
do
  if [ -e "$p" ]; then
    printf '[OK] %s -> ' "$p"
    if [ -L "$p" ]; then readlink -f "$p"; else du -sh "$p" 2>/dev/null | awk '{print $1}'; fi
  else
    printf '[MISS] %s\n' "$p"
  fi
done
```

### 4.2 集群能力验证

```bash
sudo -n true || { echo "[ERROR] need NOPASSWD sudo for rootless Docker wrapper"; exit 1; }
command -v docker >/dev/null || { echo "[ERROR] docker CLI missing"; exit 1; }
command -v dockerd >/dev/null || { echo "[ERROR] dockerd binary missing"; exit 1; }
command -v tmux >/dev/null || { echo "[ERROR] tmux missing on host"; exit 1; }
curl -x "$TBV21_PROXY_URL" -fsS --max-time 10 http://example.com >/dev/null || {
  echo "[ERROR] proxy not reachable: $TBV21_PROXY_URL"
  exit 1
}
```

如果 proxy 检查失败，先切换到当前 worker 可用代理，再重新加载 env：

```bash
export TBV21_PROXY_URL=http://<your-proxy-host>:<port>
export HTTP_PROXY="$TBV21_PROXY_URL"
export HTTPS_PROXY="$TBV21_PROXY_URL"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
source ./env.sh
cat docker-cli/config.json
curl -x "$TBV21_PROXY_URL" -fsS --max-time 10 http://example.com >/dev/null
```

如果 `sudo -n true` 失败，本文的 rootless Docker wrapper 不能按当前方式自动启动，需要找管理员开 NOPASSWD sudo 或改用集群支持的 privileged rjob / 远程 Docker daemon 路线。

### 4.3 安装或启动 rootless Docker

```bash
df -h /tmp
# 经验底线：/tmp 至少预留 60 GiB；本次 worker /tmp 约 1.4 TiB。

if [ ! -x "$HOME/.local/bin/dockerd-rootless-launch.sh" ]; then
  bash bin/setup_worker_rootless_docker.sh
fi

bash bin/use_worker_rootless_docker.sh start
source "$TBV21_DOCKER_HOST_ENV"
docker info --format 'server={{.ServerVersion}} root={{.DockerRootDir}} driver={{.Driver}}'
docker compose version
```

期望：

```text
driver=fuse-overlayfs
root=/tmp/tbv21-rootless-docker-<user>
```

如果 root 指到 GPFS 上，改：

```bash
export TBV21_ROOTLESS_DOCKER_DATA_ROOT=/tmp/tbv21-rootless-docker-${USER}
bash bin/use_worker_rootless_docker.sh start
```

### 4.4 容器联网 smoke

```bash
bash bin/use_worker_rootless_docker.sh smoke
```

必须看到 `hello-world` 和 `alpine wget http://example.com` 成功。否则不要跑 Harbor。

### 4.5 预拉镜像

```bash
bash bin/prepull_tbv21_images.sh "$TBV21_TASKS_DIR"
```

必须看到：

```text
prepull summary: total=89 ok=89 failed=0
```

### 4.6 启动 SGLang

```bash
export TBV21_GPU_IDS=0,1,2,3
bash bin/start_sglang.sh "$TBV21_GPU_IDS"
curl --noproxy '*' -fsS --max-time 10 "http://${TBV21_SGLANG_HOST}:${TBV21_SGLANG_PORT}/v1/models"
```

如果新 worker GPU 数不同：

- 4 卡：`TBV21_GPU_IDS=0,1,2,3`，`TP_SIZE=4` 自动推断。
- 8 卡但只想用 4 卡：仍设 `0,1,2,3`。
- 1 卡小模型测试：改模型和 `TBV21_GPU_IDS=0`，但 Qwen3-8B 这个服务在当前验证里使用 4 卡。

如果 SGLang CUDA OOM，先降低静态显存占比再重启：

```bash
export TBV21_SGLANG_MEM_FRACTION=0.50
bash bin/start_sglang.sh "$TBV21_GPU_IDS"
```

### 4.7 单任务 smoke

```bash
bash bin/run_one_task_eval_qwen3_8b.sh regex-chess
```

单任务能创建容器、调用模型、写 `jobs/one_task.../result.json` 后，再跑 full eval。

### 4.8 完整 eval

进入 `gpu:4.0` 跑本文后续 full eval 命令。如果新 worker 没有 `gpu` tmux session，先创建：

```bash
tmux has-session -t gpu 2>/dev/null || tmux new-session -d -s gpu -n shell
while [ "$(tmux list-windows -t gpu | wc -l | awk '{print $1}')" -le 4 ]; do
  tmux new-window -t gpu
done
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_current_command}'
```

更简单的方式是不用 `gpu:4.0` 约定，改用专门 session：

```bash
tmux kill-session -t tbv21_full_eval 2>/dev/null || true
tmux new-session -d -s tbv21_full_eval -n eval
```

后续所有 `tmux send-keys -t gpu:4.0` 可替换为 `tmux send-keys -t tbv21_full_eval:0.0`。

---

## 5. 前置条件

### 5.1 Worker 与路径

默认路径：

```bash
export TBV21_HOME=/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
cd "$TBV21_HOME"
```

必须存在：

```bash
test -d "$TBV21_HOME"
test -d "$TBV21_HOME/datasets/terminal-bench-2-1/tasks"
test -r "$(readlink -f "$TBV21_HOME/models/qwen3-8b-rl-iter215")/config.json"
test -x "$TBV21_HOME/bin/harbor"
test -x "$TBV21_HOME/bin/python_sglang"
test -x "$TBV21_HOME/runtime/tmux_runtime/bin/tmux"
test -x "$TBV21_HOME/runtime/tools/bin/uv"
test -d "$TBV21_HOME/runtime/wheelhouse"
test -x "$TBV21_HOME/docker-cli-plugins/docker-compose"
```

如果 `bin/harbor` 或 `bin/python_sglang` 是 shared conda wrapper，必须确认 wrapper 目标可执行：

```bash
bin/harbor --version
bin/python_sglang - <<'PY'
import importlib.util, torch
print("cuda_count", torch.cuda.device_count())
for name in ["sglang", "openai", "transformers"]:
    print(name, bool(importlib.util.find_spec(name)))
PY
```

### 5.2 tmux 页面约定

本次实际使用：

- `gpu:4.0`：完整 Harbor eval 主页面。
- `tbv21_rootless_docker:0.0`：rootless Docker daemon 页面，由脚本自动创建。
- `tbv21_sglang_30000:0.0`：SGLang 页面，由脚本自动创建或复用。

查看页面：

```bash
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} active=#{pane_active} cmd=#{pane_current_command} title=#{pane_title}'
```

如果没有 `gpu:4.0`：

```bash
tmux has-session -t gpu 2>/dev/null || tmux new-session -d -s gpu -n shell
while [ "$(tmux list-windows -t gpu | wc -l | awk '{print $1}')" -le 4 ]; do
  tmux new-window -t gpu
done
```

或者使用 `tbv21_full_eval:0.0` 替代文档中的 `gpu:4.0`。

### 5.3 代理

PJLab worker 需要代理。默认代理：

```bash
export TBV21_PROXY_URL=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128
export HTTP_PROXY="$TBV21_PROXY_URL"
export HTTPS_PROXY="$TBV21_PROXY_URL"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export NO_PROXY='localhost,127.0.0.1,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,.pjlab.local,.svc'
export no_proxy="$NO_PROXY"
source ./env.sh
cat docker-cli/config.json
```

必须绕过本地 SGLang：

```bash
curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:30000/v1/models
```

## 6. 环境变量与默认配置

进入工作目录并加载 `env.sh`：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
source ./env.sh
```

重要顺序规则：

- `source ./env.sh` 只负责设置默认路径、代理和 Docker CLI config，不保证 Docker daemon 已经启动。
- 任何 `docker info`、`docker pull`、`harbor run -e docker` 之前，都要先执行 `bash bin/use_worker_rootless_docker.sh start`。
- 启动脚本会写 `state/docker_host.env`；启动后再执行 `source "$TBV21_DOCKER_HOST_ENV"`，确保当前 shell 的 `DOCKER_HOST` 和实际 rootless daemon 对齐。

本次关键默认值来自 `env.sh`：

```bash
export TBV21_HOME="${TBV21_HOME:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export TBV21_DATASET_DIR="${TBV21_DATASET_DIR:-${TBV21_HOME}/datasets/terminal-bench-2-1}"
export TBV21_TASKS_DIR="${TBV21_TASKS_DIR:-${TBV21_DATASET_DIR}/tasks}"
export TBV21_JOBS_DIR="${TBV21_JOBS_DIR:-${TBV21_HOME}/jobs}"
export TBV21_LOG_DIR="${TBV21_LOG_DIR:-${TBV21_HOME}/logs}"
export TBV21_STATE_DIR="${TBV21_STATE_DIR:-${TBV21_HOME}/state}"

export TBV21_PROXY_URL="${TBV21_PROXY_URL:-${HTTP_PROXY:-http://httpproxy-headless.kubebrain.svc.pjlab.local:3128}}"
export HTTP_PROXY="${HTTP_PROXY:-${TBV21_PROXY_URL}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${TBV21_PROXY_URL}}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,.pjlab.local,.svc}"

export TBV21_ROOTLESS_DOCKER_DATA_ROOT="${TBV21_ROOTLESS_DOCKER_DATA_ROOT:-/tmp/tbv21-rootless-docker-${USER:-user}}"
export TBV21_ROOTLESS_DOCKER_RUNTIME_DIR="${TBV21_ROOTLESS_DOCKER_RUNTIME_DIR:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}}"
export TBV21_ROOTLESS_DOCKER_HOST="${TBV21_ROOTLESS_DOCKER_HOST:-unix://${TBV21_ROOTLESS_DOCKER_RUNTIME_DIR}/docker.sock}"
export TBV21_DOCKER_REGISTRY_MIRRORS="${TBV21_DOCKER_REGISTRY_MIRRORS:-https://docker.1ms.run}"
export TBV21_DOCKERHUB_MIRROR_PREFIX="${TBV21_DOCKERHUB_MIRROR_PREFIX:-docker.1ms.run}"
export TBV21_PREPULL_IMAGES="${TBV21_PREPULL_IMAGES:-1}"
export TBV21_PREPULL_RETRIES="${TBV21_PREPULL_RETRIES:-5}"
export TBV21_PREPULL_TIMEOUT_S="${TBV21_PREPULL_TIMEOUT_S:-900}"
export TBV21_PREPULL_REQUIRED="${TBV21_PREPULL_REQUIRED:-1}"
export DOCKER_HOST="${DOCKER_HOST:-${TBV21_ROOTLESS_DOCKER_HOST}}"
export DOCKER_CONFIG="${DOCKER_CONFIG:-${TBV21_DOCKER_CONFIG_DIR}}"
export TBV21_DOCKER_CONFIG_REWRITE="${TBV21_DOCKER_CONFIG_REWRITE:-1}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:30000/v1}"
export TBV21_MODEL_NAME="${TBV21_MODEL_NAME:-qwen3-8b-rl-iter215}"
export TBV21_MODEL_PATH="${TBV21_MODEL_PATH:-${TBV21_HOME}/models/qwen3-8b-rl-iter215}"
export TBV21_MODEL_MAX_INPUT_TOKENS="${TBV21_MODEL_MAX_INPUT_TOKENS:-40960}"
export TBV21_MODEL_MAX_OUTPUT_TOKENS="${TBV21_MODEL_MAX_OUTPUT_TOKENS:-4096}"
export TBV21_SGLANG_HOST="${TBV21_SGLANG_HOST:-127.0.0.1}"
export TBV21_SGLANG_PORT="${TBV21_SGLANG_PORT:-30000}"
export TBV21_SGLANG_MEM_FRACTION="${TBV21_SGLANG_MEM_FRACTION:-0.70}"
export TBV21_SGLANG_DISABLE_CUDA_GRAPH="${TBV21_SGLANG_DISABLE_CUDA_GRAPH:-1}"
```

`env.sh` 还会默认重写 Docker CLI proxy config，让 `docker build/run` 自动给容器注入代理。若当前 shell 里已有 `HTTP_PROXY`，它优先于默认 `TBV21_PROXY_URL`：

```json
{
  "proxies": {
    "default": {
      "httpProxy": "${HTTP_PROXY}",
      "httpsProxy": "${HTTPS_PROXY}",
      "noProxy": "${NO_PROXY}"
    }
  }
}
```

## 7. Rootless Docker

### 7.1 首次安装 rootless Docker

如果 `~/.local/bin/dockerd-rootless-launch.sh` 不存在，执行：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
source ./env.sh
bash bin/setup_worker_rootless_docker.sh
```

该脚本要求：

- 普通用户运行，不要 `sudo bash` 整个脚本。
- 当前用户有 `NOPASSWD sudo`。
- pod 内已有 `docker` CLI 和 `dockerd` 二进制。
- 能通过代理下载 docker-rootless-extras 和 `slirp4netns`。

安装脚本关键逻辑：

```bash
sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    fuse-overlayfs uidmap curl ca-certificates

ROOTLESS_URL="https://download.docker.com/linux/static/stable/x86_64/docker-rootless-extras-${DOCKER_VER}.tgz"
curl -fsSL -o rootless-extras.tgz "$ROOTLESS_URL"

curl -fsSL -o "$HOME/.local/bin/slirp4netns" \
    https://github.com/rootless-containers/slirp4netns/releases/download/v1.3.2/slirp4netns-x86_64

sed -i 's|--propagation=rslave|--propagation=private|' "$HOME/.local/bin/dockerd-rootless.sh"
```

K8s pod 里 rootlesskit 默认 `propagation=rslave` 可能被内核拒绝，所以安装脚本改成 `private`。启动 wrapper 还会用私有 mount namespace 处理 `/proc`、`/sys` 覆盖和 `/dev/net/tun`：

```bash
exec sudo -n -E unshare -m -- bash -c '
  ...
  mount -t devtmpfs -o nosuid none /run/host-devtmpfs 2>/dev/null
  if [ -e /run/host-devtmpfs/net/tun ]; then
    mkdir -p /dev/net
    mount --bind /run/host-devtmpfs/net/tun /dev/net/tun 2>/dev/null \
      || { touch /dev/net/tun; mount --bind /run/host-devtmpfs/net/tun /dev/net/tun; }
  fi
  exec runuser -u '"$USER"' -- env \
    HOME='"$HOME"' \
    XDG_RUNTIME_DIR=/run/user/'"$(id -u)"' \
    DOCKERD_ROOTLESS_ROOTLESSKIT_NET=slirp4netns \
    DOCKERD_ROOTLESS_ROOTLESSKIT_PORT_DRIVER=builtin \
    '"$HOME"'/.local/bin/dockerd-rootless.sh
'
```

### 7.2 启动 rootless Docker

常规启动：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
source ./env.sh
df -h /tmp
# 经验底线：/tmp 至少预留 60 GiB；本次 worker /tmp 约 1.4 TiB。
bash bin/use_worker_rootless_docker.sh start
source "$TBV21_DOCKER_HOST_ENV"
```

期望：

```bash
echo "$DOCKER_HOST"
# 本机示例：unix:///run/user/10250/docker.sock；新环境以 id -u 输出为准

docker info --format 'ServerVersion={{.ServerVersion}} Driver={{.Driver}} DockerRootDir={{.DockerRootDir}}'
# ServerVersion=26.1.3 Driver=fuse-overlayfs DockerRootDir=/tmp/tbv21-rootless-docker-zhangshaoang
```

启动脚本会写 rootless daemon config：

```python
config = {
    "storage-driver": "fuse-overlayfs",
    "data-root": sys.argv[2],
}

mirrors = [
    item.strip()
    for item in os.environ.get("TBV21_DOCKER_REGISTRY_MIRRORS", "").split(",")
    if item.strip()
]
if mirrors:
    config["registry-mirrors"] = mirrors

proxies = {}
if os.environ.get("HTTP_PROXY"):
    proxies["http-proxy"] = os.environ["HTTP_PROXY"]
if os.environ.get("HTTPS_PROXY"):
    proxies["https-proxy"] = os.environ["HTTPS_PROXY"]
if os.environ.get("NO_PROXY"):
    proxies["no-proxy"] = os.environ["NO_PROXY"]
if proxies:
    config["proxies"] = proxies
```

### 7.3 Docker smoke test

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
source ./env.sh
bash bin/use_worker_rootless_docker.sh smoke
```

期望至少：

```text
[OK] rootless Docker ready
[INFO] docker run hello-world
[INFO] docker run alpine wget example.com
[OK] rootless Docker smoke passed
```

如果 `docker version` 报 EOF，但 `docker info` 正常，先不要误判；当前环境中固定 `DOCKER_API_VERSION=1.45` 是为了绕过 dockerd 26.1.3 在该 rootless 组合下 `/version` 协商 panic 的已知问题。

## 8. 预拉 TBv2.1 images

Harbor full eval 不应该边跑边等 Docker Hub。必须先预拉。

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
source ./env.sh
bash bin/use_worker_rootless_docker.sh start
source "$TBV21_DOCKER_HOST_ENV"
bash bin/prepull_tbv21_images.sh "$TBV21_TASKS_DIR"
```

本次验证结果：

```text
prepull summary: total=89 ok=89 failed=0
```

预拉脚本做三件事：

1. 从每个 `task.toml` 解析 `docker_image`。
2. 对 89 个 unique images 逐个 `docker pull`。
3. 如果官方 Docker Hub 拉取失败，则尝试 `docker.1ms.run/<image>`，成功后 tag 回原名。

关键代码：

```python
for task_toml in sorted(tasks_dir.glob("*/task.toml")):
    text = task_toml.read_text(errors="replace")
    match = re.search(r'^\s*docker_image\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if match:
        rows.append((match.group(1), task_toml.parent.name))
```

```bash
if timeout "${PULL_TIMEOUT_S}" docker pull "${image}"; then
  log "[OK] pulled: ${image}"
  return 0
fi

if mirror_image="$(mirror_image_for "${image}")"; then
  log "[INFO] mirror pull attempt ${attempt}/${RETRIES}: ${mirror_image}"
  if timeout "${PULL_TIMEOUT_S}" docker pull "${mirror_image}"; then
    docker tag "${mirror_image}" "${image}"
    log "[OK] pulled via mirror and tagged: ${mirror_image} -> ${image}"
    return 0
  fi
fi
```

失败处理：

- `failed=0`：继续。
- `failed>0` 且 `TBV21_PREPULL_REQUIRED=1`：不要启动 full eval，先解决镜像拉取。
- 常见修复：确认代理、确认 rootless Docker daemon config 里的 registry mirror、单独 `docker pull` 失败镜像。

## 9. 启动或复用 SGLang

启动：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
source ./env.sh
export TBV21_GPU_IDS=0,1,2,3
bash bin/start_sglang.sh "$TBV21_GPU_IDS"
```

SGLang 实际命令：

```bash
python -m sglang.launch_server \
  --model-path /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1/models/qwen3-8b-rl-iter215 \
  --served-model-name qwen3-8b-rl-iter215 \
  --host 127.0.0.1 \
  --port 30000 \
  --tp-size 4 \
  --trust-remote-code \
  --mem-fraction-static 0.70 \
  --disable-cuda-graph
```

脚本会先探测健康的已有服务：

```bash
if curl --noproxy '*' -fsS --max-time 5 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[OK] sglang already ready port=${PORT}"
  curl --noproxy '*' -fsS --max-time 5 "http://${HOST}:${PORT}/v1/models" || true
  echo
  exit 0
fi
```

这是重要修复：不要因为 pid 文件缺失就杀掉健康的 SGLang。健康标准以 `/v1/models` 为准。

期望输出：

```json
{"object":"list","data":[{"id":"qwen3-8b-rl-iter215","object":"model","owned_by":"sglang","root":"qwen3-8b-rl-iter215","max_model_len":40960}]}
```

查看日志：

```bash
tail -f logs/sglang/sglang_qwen3-8b-rl-iter215_30000.log
```

## 10. Harbor full eval

### 10.1 推荐一键命令

推荐在专用 tmux session 里运行。当前 worker 也可以使用既有的 `gpu:4.0`，但新 worker 默认用 `tbv21_full_eval:0.0`，避免 window 编号不存在。

```bash
TARGET_PANE="${TARGET_PANE:-tbv21_full_eval:0.0}"
tmux has-session -t "${TARGET_PANE%%:*}" 2>/dev/null || tmux new-session -d -s "${TARGET_PANE%%:*}" -n eval
tmux send-keys -t "$TARGET_PANE" "cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1" C-m
tmux send-keys -t "$TARGET_PANE" "source ./env.sh" C-m
tmux send-keys -t "$TARGET_PANE" "export TBV21_FULL_EVAL_JOB_NAME=full_eval_tbv21_qwen3-8b-rl-iter215_\$(date +%Y%m%d_%H%M%S)" C-m
tmux send-keys -t "$TARGET_PANE" "export TBV21_GPU_IDS=0,1,2,3" C-m
tmux send-keys -t "$TARGET_PANE" "export TBV21_FULL_EVAL_CONCURRENCY=2" C-m
tmux send-keys -t "$TARGET_PANE" "export TBV21_FULL_EVAL_MAX_RETRIES=1" C-m
tmux send-keys -t "$TARGET_PANE" "export TBV21_FULL_EVAL_TIMEOUT_MULTIPLIER=1.0" C-m
tmux send-keys -t "$TARGET_PANE" "bash bin/run_full_eval_qwen3_8b.sh" C-m
```

如果不是通过 `tmux send-keys`，也可以直接在 `gpu:4.0` 交互 shell 里复制：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
source ./env.sh
export TBV21_FULL_EVAL_JOB_NAME=full_eval_tbv21_qwen3-8b-rl-iter215_$(date +%Y%m%d_%H%M%S)
export TBV21_GPU_IDS=0,1,2,3
export TBV21_FULL_EVAL_CONCURRENCY=2
export TBV21_FULL_EVAL_MAX_RETRIES=1
export TBV21_FULL_EVAL_TIMEOUT_MULTIPLIER=1.0
bash bin/run_full_eval_qwen3_8b.sh
```

启动后 5-10 秒检查日志确实生成，避免 tmux target 错误导致“以为启动了但没跑”：

```bash
JOB="<你设置的 TBV21_FULL_EVAL_JOB_NAME>"
sleep 10
test -s "logs/${JOB}.log" || {
  echo "[ERROR] Harbor log was not created or is empty: logs/${JOB}.log"
  tmux capture-pane -t "${TARGET_PANE:-tbv21_full_eval:0.0}" -p -S -80 || true
  exit 1
}
tail -n 80 "logs/${JOB}.log"
```

### 10.2 `run_full_eval_qwen3_8b.sh` 做了什么

顺序：

1. 创建 log/state/job 目录。
2. 检查 tasks、model、tmux runtime、uv、wheelhouse。
3. 写 `/opt/tbv21-apt/apt.conf` 对应的宿主 runtime 文件。
4. 启动 rootless Docker。
5. `docker info`、`docker compose version` 自检。
6. 预拉 TBv2.1 images。
7. 启动或复用 SGLang。
8. 生成 Harbor config。
9. 自检 Harbor config。
10. `harbor run --config ...`。

关键 APT proxy 文件：

```bash
cat > "${APT_RUNTIME_DIR}/apt.conf" <<EOF
Acquire::http::Proxy "${HTTP_PROXY}";
Acquire::https::Proxy "${HTTPS_PROXY}";
Acquire::Retries "3";
Acquire::http::Timeout "30";
Acquire::https::Timeout "30";
EOF
```

关键 Harbor config 中的 container env：

```python
env = {
    "PATH": env_container_path,
    "APT_CONFIG": "/opt/tbv21-apt/apt.conf",
    "DEBIAN_FRONTEND": "noninteractive",
    "TZ": "Etc/UTC",
    "UV_LINK_MODE": "copy",
    "UV_PYTHON_INSTALL_DIR": "/tmp/tbv21-uv-python",
    "UV_CACHE_DIR": "/tmp/tbv21-uv-cache",
    "UV_FIND_LINKS": "/opt/tbv21-wheelhouse",
    "UV_INDEX_URL": "http://mirrors.h.pjlab.org.cn/pypi/simple/",
    "UV_EXTRA_INDEX_URL": "http://pypi.i.h.pjlab.org.cn/brain/dev/+simple",
    "UV_TRUSTED_HOST": "mirrors.h.pjlab.org.cn pypi.i.h.pjlab.org.cn",
}
for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"):
    if os.environ.get(key):
        env[key] = os.environ[key]
```

关键 bind mounts：

```json
[
  {
    "type": "bind",
    "source": "/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1/runtime/tmux_runtime",
    "target": "/opt/tbv21-tmux",
    "read_only": true,
    "bind": {"create_host_path": false}
  },
  {
    "type": "bind",
    "source": "/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1/runtime/apt",
    "target": "/opt/tbv21-apt",
    "read_only": true,
    "bind": {"create_host_path": false}
  },
  {
    "type": "bind",
    "source": "/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1/runtime/tools",
    "target": "/opt/tbv21-tools",
    "read_only": true,
    "bind": {"create_host_path": false}
  },
  {
    "type": "bind",
    "source": "/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1/runtime/wheelhouse",
    "target": "/opt/tbv21-wheelhouse",
    "read_only": true,
    "bind": {"create_host_path": false}
  }
]
```

关键 agent config：

```python
"agents": [
    {
        "name": "terminus-2",
        "model_name": f"openai/{os.environ['TBV21_MODEL_NAME']}",
        "kwargs": {
            "api_base": f"http://{os.environ['TBV21_SGLANG_HOST']}:{os.environ['TBV21_SGLANG_PORT']}/v1",
            "record_terminal_session": False,
            "model_info": {
                "max_input_tokens": int(os.environ["TBV21_MODEL_MAX_INPUT_TOKENS"]),
                "max_output_tokens": int(os.environ["TBV21_MODEL_MAX_OUTPUT_TOKENS"]),
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            },
        },
        "env": {
            "OPENAI_API_KEY": "dummy",
            "OPENAI_BASE_URL": f"http://{os.environ['TBV21_SGLANG_HOST']}:{os.environ['TBV21_SGLANG_PORT']}/v1",
            "PATH": agent_container_path,
            "APT_CONFIG": "/opt/tbv21-apt/apt.conf",
            "DEBIAN_FRONTEND": "noninteractive",
            "TZ": "Etc/UTC",
        },
    }
]
```

`model_info` 是必须项。否则 LiteLLM 可能不认识本地模型上下文，出现 unknown model/context fallback 类问题。

### 10.3 Harbor config 自检

`run_full_eval_qwen3_8b.sh` 在启动 Harbor 前会检查：

```python
required_env = {
    "DEBIAN_FRONTEND": "noninteractive",
    "TZ": "Etc/UTC",
    "APT_CONFIG": "/opt/tbv21-apt/apt.conf",
}
missing = []
for key, expected in required_env.items():
    if env.get(key) != expected:
        missing.append(f"environment.env.{key}={env.get(key)!r}")
    if agent_env.get(key) != expected:
        missing.append(f"agents[0].env.{key}={agent_env.get(key)!r}")
for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
    if key not in env:
        missing.append(f"environment.env.{key}=<missing>")
for key in ("max_input_tokens", "max_output_tokens"):
    if int(model_info.get(key) or 0) <= 0:
        missing.append(f"agents[0].kwargs.model_info.{key}={model_info.get(key)!r}")
if missing:
    raise SystemExit("invalid Harbor config: " + "; ".join(missing))
print("[OK] Harbor config has proxy env, noninteractive apt env, and model_info")
```

期望输出：

```text
[OK] Harbor config has proxy env, noninteractive apt env, and model_info
```

如果自检失败，不要手动启动 Harbor；修 config 生成逻辑。

## 11. 无人值守监控

### 11.1 找到当前 job

`run_full_eval_qwen3_8b.sh` 会写：

```bash
state/${JOB_NAME}.env
logs/${JOB_NAME}.log
state/${JOB_NAME}.config.json
jobs/${JOB_NAME}
```

本次：

```bash
JOB=full_eval_tbv21_qwen3-8b-rl-iter215_ready_20260701_215702
```

### 11.2 5 分钟轮询脚本

在另一个 shell 运行：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
source ./env.sh
source "$TBV21_DOCKER_HOST_ENV"
JOB=full_eval_tbv21_qwen3-8b-rl-iter215_ready_20260701_215702

while true; do
  echo "=== $(date '+%F %T') ==="
  python3 - <<PY
from pathlib import Path
import json, collections, time
job=Path("jobs/${JOB}")
res=job/"result.json"
finished=None; total=None
if res.exists():
    d=json.loads(res.read_text())
    finished=d.get("finished_at")
    total=d.get("n_total_trials")
results=list(job.glob("*/result.json"))
errs=collections.Counter()
rewards=[]
for p in results:
    d=json.loads(p.read_text())
    exc=d.get("exception_info")
    if exc:
        errs[exc.get("exception_type") or "unknown"] += 1
    r=((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if r is not None:
        rewards.append(r)
print("finished_at=", finished, "done=", len(results), "/", total, "errors=", dict(errs), "reward_sum=", sum(rewards), "mean_present=", (sum(rewards)/len(rewards) if rewards else None))
for p in sorted(results, key=lambda x:x.stat().st_mtime)[-8:]:
    d=json.loads(p.read_text())
    exc=d.get("exception_info")
    r=((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    print(time.strftime("%H:%M:%S", time.localtime(p.stat().st_mtime)), p.parent.name, "reward", r, "exc", (exc or {}).get("exception_type"))
PY
  echo "containers:"
  docker ps --format '{{.Names}} {{.Status}} {{.Image}}' || true
  tail -n 5 "logs/${JOB}.log" 2>/dev/null || true
  curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:30000/v1/models >/dev/null && echo 'sglang: ok' || echo 'sglang: BAD'
  if python3 - <<PY
import json
from pathlib import Path
p=Path("jobs/${JOB}/result.json")
if not p.exists():
    raise SystemExit(1)
d=json.loads(p.read_text())
raise SystemExit(0 if d.get("finished_at") else 1)
PY
  then
    break
  fi
  sleep 300
done
```

### 11.3 正常推进的判断

正常现象：

- `done` 持续增加，或当前 active container 的运行时间还没有超过该 task timeout。
- `docker ps` 有 1-2 个 task containers，随着任务完成不断换名字。
- `sglang: ok`。
- `AgentTimeoutError` 数量增加，但 container 释放后新任务继续启动。

本次收尾阶段的正常例子：

```text
finished_at= None done= 86 / 89 errors= {'AgentTimeoutError': 20}
containers:
sqlite-db-truncate__dfvnfsq-main-1 Up About a minute alexgshaw/sqlite-db-truncate:20251031
query-optimize__yfuelxg-main-1 Up 3 minutes alexgshaw/query-optimize:20251031
sglang: ok
```

最终：

```text
finished_at= 2026-07-02T07:22:00.608529 done= 89 / 89 errors= {'AgentTimeoutError': 20} reward_sum= 2.0
containers:
sglang: ok
```

### 11.4 什么时候需要介入

只在 infra 确认坏掉时介入：

- `curl --noproxy '*' http://127.0.0.1:30000/v1/models` 失败，且 SGLang tmux/log 显示服务退出。
- `docker info` 失败或 `_ping` 不通。
- `docker ps` 卡死。
- Harbor 主进程退出，但 `jobs/$JOB/result.json` 没有 `finished_at`。
- 某个 container 远超 task timeout 仍未释放，并且 Harbor 没有写该 task result。

不要因为这些情况介入：

- 单个 task 跑 15、30、60 分钟。TBv2.1 有任务 timeout 为 `3600s`，`build-pov-ray` 这类还可能更久。
- `AgentTimeoutError`。这是模型/task 结果，不是基础设施失败。
- 主 log 有 `Unclosed client session` warning。本次完整 eval 里出现过，但没有阻断。

## 12. 结果解析

完成后运行：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
JOB=full_eval_tbv21_qwen3-8b-rl-iter215_ready_20260701_215702

python3 - <<PY
from pathlib import Path
import json, collections
job=Path("jobs/${JOB}")
res=json.loads((job/"result.json").read_text())
results=[]
for p in sorted(job.glob("*/result.json")):
    d=json.loads(p.read_text())
    exc=d.get("exception_info")
    reward=((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    results.append((p.parent.name, reward, exc, d))
errs=collections.Counter((exc or {}).get("exception_type") for _,_,exc,_ in results if exc)
positives=[(name,reward) for name,reward,exc,_ in results if reward and reward>0]
missing=[name for name,reward,exc,_ in results if reward is None]
reward_sum=sum(r or 0 for _,r,_,_ in results)
print("job_dir", job)
print("started_at", res.get("started_at"))
print("finished_at", res.get("finished_at"))
print("n_total_trials", res.get("n_total_trials"))
print("n_result_files", len(results))
print("error_counts", dict(errs))
print("reward_sum", reward_sum)
print("mean_all_total", reward_sum / (res.get("n_total_trials") or len(results)))
print("missing_reward", missing)
print("positive_tasks")
for name,reward in positives:
    print(name, reward)
print("timeout_tasks")
for name,reward,exc,d in results:
    if exc:
        print(name, (exc or {}).get("exception_type"), (exc or {}).get("exception_message","")[:160].replace("\\n"," "))
PY
```

本次输出核心：

```text
started_at 2026-07-01T22:00:13.674888
finished_at 2026-07-02T07:22:00.608529
n_total_trials 89
n_result_files 89
error_counts {'AgentTimeoutError': 20}
reward_sum 2.0
mean_all_total 0.02247191011235955
positive_tasks
configure-git-webserver__zVAaSVv 1.0
hf-model-inference__72rnD2H 1.0
```

注意两个分数口径：

- Harbor `result.json` 的聚合分数：`2.0 / 89 = 0.0224719101`。
- 只对有 reward 字段的 88 个 result 算 mean：`2.0 / 88 = 0.0227272727`。

报告时统一使用 Harbor 聚合口径，也就是 `n_total_trials` 作为分母。

## 13. 收尾检查

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
source ./env.sh
source "$TBV21_DOCKER_HOST_ENV"

docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' | head -50
curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:30000/v1/models
ps -ef | grep -E 'sglang|harbor run|run_full_eval_qwen3_8b' | grep -v grep || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

本次完成后：

- `docker ps -a` 没有 Harbor task containers。
- SGLang 仍在运行。
- Harbor 主流程已结束。

如果要释放 GPU：

```bash
tmux kill-session -t tbv21_sglang_30000
```

如果要停止 rootless Docker：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
source ./env.sh
bash bin/use_worker_rootless_docker.sh stop
```

默认建议：如果还要继续 eval，不要停 SGLang，不要 prune images。

## 14. 单任务 smoke eval

在完整 eval 前可以先跑一个短任务：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
source ./env.sh
export TBV21_GPU_IDS=0,1,2,3
bash bin/run_one_task_eval_qwen3_8b.sh regex-chess
```

`run_one_task_eval_qwen3_8b.sh` 与 full eval 使用同一套 Docker、SGLang、代理、APT、model_info 配置，只是：

```python
"n_concurrent_trials": 1,
"retry": {"max_retries": 0},
"datasets": [
    {
        "path": os.environ["TBV21_TASKS_DIR"],
        "task_names": [task],
    }
],
```

如果单任务报 model/context 或 tzdata/apt hang，先修通它，再跑 full eval。

## 15. 常见问题与处理


### 15.1 本次真实问题复盘总表

| 问题 | 现象 | 根因 | 解决方式 | 当前状态 |
|---|---|---|---|---|
| 普通 nested dockerd 不适合当前 worker | `docker run` 失败，曾出现 `/sys/fs/cgroup/... read-only file system` 类错误 | 当前 GPU worker 是受限 pod，不能假设 systemd Docker 或 privileged Docker-in-Docker | 改为 rootless Docker：`dockerd-rootless` + `rootlesskit` + `slirp4netns` + `fuse-overlayfs` | 已解决 |
| Docker data root 放共享文件系统不稳 | image layer unpack 可能出现 `Lchown ... operation not permitted` | GPFS/共享存储对 overlay layer 的 owner/chown 操作不完全兼容 | rootless Docker data root 改到 `/tmp/tbv21-rootless-docker-$USER` | 已解决 |
| Docker Hub 拉取慢或 TLS timeout | `docker pull` 过程中可能 `TLS handshake timeout` / `context deadline exceeded` | 当前集群到 Docker Hub 网络不稳定 | rootless daemon 配 `registry-mirrors`；eval 前 `prepull_tbv21_images.sh` 预拉 89 个 images；失败时走 `docker.1ms.run` fallback | 已解决，本次 `failed=0` |
| task container 内 apt/pip/download 不通 | 容器内安装依赖或下载资源慢、超时 | 只给宿主 shell 配代理不够，容器内不一定继承；APT 也不一定读取普通 env | Docker CLI proxy、Harbor environment env、`APT_CONFIG=/opt/tbv21-apt/apt.conf`、uv/PyPI mirror、wheelhouse 四层同时配置 | 已解决基础网络问题 |
| `tzdata` 等包交互卡住 | task 看似运行但长时间不动 | apt 安装包等待交互输入 | Harbor config 注入 `DEBIAN_FRONTEND=noninteractive` 和 `TZ=Etc/UTC` | 已解决 |
| LiteLLM 不认识本地模型 | 本地 OpenAI-compatible model 可能被当作 unknown model，context/cost 信息缺失 | 只传 `model_name=openai/...` 不足以让 LiteLLM 知道上下文长度 | Harbor agent config 加 `model_info.max_input_tokens=40960`、`max_output_tokens=4096`、cost=0 | 已解决 |
| SGLang pid 文件不可靠 | pid 文件缺失时可能误杀健康服务 | tmux 启动的服务不一定留下可用 pid 文件 | `start_sglang.sh` 先探测 `/v1/models`；健康则复用，不杀 session | 已解决 |
| task 自身外部数据慢 | 最终有 `AgentTimeoutError` | 某些 task 需要下载外部 benchmark/dataset 或模型输出无法完成任务 | 不修改 task；通过 timeout 归入 eval 结果。基础设施只保证容器、代理、SGLang、Harbor 正常 | 属于 eval 结果，不是 infra 故障 |
| rootless Docker cgroup 日志噪声 | daemon log 中有 `cgroups: cgroup deleted`、容器 SIGTERM 后 force kill | rootless Docker 在当前 pod 中 `Cgroup Driver: none`，任务结束清理时会有日志噪声 | 以 `docker ps -a` 无残留、Harbor result 完成、SGLang 健康作为判断，不把该日志单独视为失败 | 可接受 |

这张表里的处理都已经固化到脚本或 runbook 中，不需要后续 agent 再手工给单个 task 打补丁。

### 15.2 Docker Hub TLS timeout / pull 慢

现象：

```text
net/http: TLS handshake timeout
context deadline exceeded
```

处理：

```bash
source ./env.sh
source "$TBV21_DOCKER_HOST_ENV"
bash bin/prepull_tbv21_images.sh "$TBV21_TASKS_DIR"
```

确认 `TBV21_DOCKERHUB_MIRROR_PREFIX=docker.1ms.run`，脚本会 fallback：

```text
mirror pull attempt ... docker.1ms.run/alexgshaw/...
pulled via mirror and tagged
```

### 15.3 task container 内 apt 卡住或 `tzdata` 交互

必须同时注入：

```bash
DEBIAN_FRONTEND=noninteractive
TZ=Etc/UTC
APT_CONFIG=/opt/tbv21-apt/apt.conf
HTTP_PROXY / HTTPS_PROXY / NO_PROXY
```

本次 `caffe-cifar-10` 之前会卡 `tzdata`，加入上述 env 后没有再卡交互。它后来超时是 CIFAR 外部下载/任务本身慢，属于 task-level timeout。

### 15.4 LiteLLM unknown model / context fallback

Harbor agent config 必须包含：

```json
"model_info": {
  "max_input_tokens": 40960,
  "max_output_tokens": 4096,
  "input_cost_per_token": 0,
  "output_cost_per_token": 0
}
```

不要只写 `model_name=openai/qwen3-8b-rl-iter215`。

### 15.5 SGLang pid 文件缺失但服务健康

不要直接杀 tmux session。先探活：

```bash
curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:30000/v1/models
```

如果健康，复用。`bin/start_sglang.sh` 已按这个逻辑处理。

### 15.6 Harbor 跑完后仍有 Docker container

先确认是否 job 已 `finished_at`：

```bash
python3 - <<'PY'
import json
from pathlib import Path
JOB="full_eval_tbv21_qwen3-8b-rl-iter215_ready_20260701_215702"
p=Path("jobs")/JOB/"result.json"
print(json.loads(p.read_text()).get("finished_at") if p.exists() else None)
PY
```

如果 job 已完成但还有 task container，先查看：

```bash
docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
```

不要 prune images。只在确认是该 job 残留 task container 时按名字 `docker rm -f <container>`。

### 15.7 `docker info` 不通

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
source ./env.sh
bash bin/use_worker_rootless_docker.sh status || bash bin/use_worker_rootless_docker.sh start
source "$TBV21_DOCKER_HOST_ENV"
docker info
```

如果 tmux session 退出：

```bash
tail -n 200 logs/docker/rootless-worker-dockerd.log
tmux capture-pane -t tbv21_rootless_docker:0.0 -p -S -200
```

### 15.8 SGLang 不通

```bash
curl --noproxy '*' -v --max-time 5 http://127.0.0.1:30000/v1/models
tmux capture-pane -t tbv21_sglang_30000:0.0 -p -S -200
tail -n 240 logs/sglang/sglang_qwen3-8b-rl-iter215_30000.log
```

如果服务真的退出：

```bash
source ./env.sh
bash bin/start_sglang.sh 0,1,2,3
```

## 16. 最小可复制主流程

以下命令可以作为 agent 的主执行脚本：

```bash
set -euo pipefail

cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
source ./env.sh

# 1. 首次缺 rootless launcher 时先安装，再启 rootless Docker。
if [ ! -x "$HOME/.local/bin/dockerd-rootless-launch.sh" ]; then
  bash bin/setup_worker_rootless_docker.sh
fi
df -h /tmp
bash bin/use_worker_rootless_docker.sh start
source "$TBV21_DOCKER_HOST_ENV"
docker info --format 'server={{.ServerVersion}} root={{.DockerRootDir}} driver={{.Driver}}'

# 2. 预拉镜像。
bash bin/prepull_tbv21_images.sh "$TBV21_TASKS_DIR"

# 3. 启动或复用 SGLang。
export TBV21_GPU_IDS=0,1,2,3
bash bin/start_sglang.sh "$TBV21_GPU_IDS"
curl --noproxy '*' -fsS --max-time 10 "http://${TBV21_SGLANG_HOST}:${TBV21_SGLANG_PORT}/v1/models"
echo

# 4. 在专用 tmux session 启动完整 eval。
TARGET_PANE="${TARGET_PANE:-tbv21_full_eval:0.0}"
tmux has-session -t "${TARGET_PANE%%:*}" 2>/dev/null || tmux new-session -d -s "${TARGET_PANE%%:*}" -n eval
JOB="full_eval_tbv21_${TBV21_MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
tmux send-keys -t "$TARGET_PANE" "cd '$TBV21_HOME'" C-m
tmux send-keys -t "$TARGET_PANE" "source ./env.sh" C-m
tmux send-keys -t "$TARGET_PANE" "export TBV21_FULL_EVAL_JOB_NAME='$JOB'" C-m
tmux send-keys -t "$TARGET_PANE" "export TBV21_GPU_IDS='0,1,2,3'" C-m
tmux send-keys -t "$TARGET_PANE" "export TBV21_FULL_EVAL_CONCURRENCY=2" C-m
tmux send-keys -t "$TARGET_PANE" "export TBV21_FULL_EVAL_MAX_RETRIES=1" C-m
tmux send-keys -t "$TARGET_PANE" "export TBV21_FULL_EVAL_TIMEOUT_MULTIPLIER=1.0" C-m
tmux send-keys -t "$TARGET_PANE" "bash bin/run_full_eval_qwen3_8b.sh" C-m

sleep 10
test -s "logs/${JOB}.log" || {
  echo "[ERROR] full eval log was not created: logs/${JOB}.log" >&2
  tmux capture-pane -t "$TARGET_PANE" -p -S -120 || true
  exit 1
}

echo "[INFO] started job=$JOB"
echo "[INFO] monitor: jobs/$JOB/result.json"
```

监控直到完成：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
source ./env.sh
source "$TBV21_DOCKER_HOST_ENV"
JOB="<替换成上一步输出的 job>"

while true; do
  date '+=== %F %T ==='
  python3 - <<PY
from pathlib import Path
import json, collections
job=Path("jobs/${JOB}")
res=job/"result.json"
d=json.loads(res.read_text()) if res.exists() else {}
results=list(job.glob("*/result.json"))
errs=collections.Counter()
reward_sum=0.0
for p in results:
    r=json.loads(p.read_text())
    exc=r.get("exception_info")
    if exc:
        errs[exc.get("exception_type") or "unknown"] += 1
    reward=((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    reward_sum += reward or 0.0
print("finished_at=", d.get("finished_at"), "done=", len(results), "/", d.get("n_total_trials"), "errors=", dict(errs), "reward_sum=", reward_sum)
PY
  docker ps --format '{{.Names}} {{.Status}} {{.Image}}' || true
  tail -n 5 "logs/${JOB}.log" 2>/dev/null || true
  curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:30000/v1/models >/dev/null && echo sglang_ok || echo sglang_BAD
  python3 - <<PY && break || true
import json
from pathlib import Path
p=Path("jobs/${JOB}/result.json")
raise SystemExit(0 if p.exists() and json.loads(p.read_text()).get("finished_at") else 1)
PY
  sleep 300
done
```

## 17. 本次验证的关键证据路径

```text
/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1/jobs/full_eval_tbv21_qwen3-8b-rl-iter215_ready_20260701_215702
/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1/jobs/full_eval_tbv21_qwen3-8b-rl-iter215_ready_20260701_215702/result.json
/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1/logs/full_eval_tbv21_qwen3-8b-rl-iter215_ready_20260701_215702.log
/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1/state/full_eval_tbv21_qwen3-8b-rl-iter215_ready_20260701_215702.config.json
/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1/state/full_eval_tbv21_qwen3-8b-rl-iter215_ready_20260701_215702.env
```

这里的 `_ready_20260701_215702` 只是本次验证 run 的历史标记，不是脚本要求。新 run 使用 `full_eval_tbv21_${TBV21_MODEL_NAME}_$(date +%Y%m%d_%H%M%S)` 即可。

通用查找最近 full eval job：

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1
find jobs -maxdepth 1 -type d -name 'full_eval_tbv21_*' -printf '%T@ %p\n' | sort -n | tail -10
```

`result.json` 聚合字段：

```json
{
  "started_at": "2026-07-01T22:00:13.674888",
  "finished_at": "2026-07-02T07:22:00.608529",
  "n_total_trials": 89,
  "stats": {
    "n_trials": 89,
    "n_errors": 20
  }
}
```

实际 positive tasks：

```text
configure-git-webserver__zVAaSVv 1.0
hf-model-inference__72rnD2H 1.0
```

## 18. Ready 判据

这套流程可以认为 ready，需要同时满足：

- `bash -n env.sh bin/prepull_tbv21_images.sh bin/run_full_eval_qwen3_8b.sh bin/run_one_task_eval_qwen3_8b.sh bin/use_worker_rootless_docker.sh bin/start_sglang.sh` 全部通过。
- `docker info` 显示 rootless Docker 使用 `fuse-overlayfs` 且 root 在大盘或 `/tmp` 大空间。
- `bin/prepull_tbv21_images.sh` 输出 `failed=0`。
- `curl --noproxy '*' http://127.0.0.1:30000/v1/models` 正常返回模型。
- Harbor config 自检输出 `[OK] Harbor config has proxy env, noninteractive apt env, and model_info`。
- `jobs/$JOB/result.json` 最终存在非空 `finished_at`。
- `docker ps -a` 没有残留 task containers。

本次以上全部满足。
