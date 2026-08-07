# TBv2.1 Harbor 全量评测无人值守运行手册

## TL;DR

在受限 GPU worker（K8s pod 类环境，有 GPU 和共享存储，但没有宿主 Docker socket、没有 systemd、不能起 privileged Docker-in-Docker）上跑 Terminal-Bench v2.1 全量评测，可行路线是七步：起 rootless Docker 并固定到 `/run/user/$(id -u)/docker.sock`；在 Docker daemon 和 task container 两层都注入代理；评测前预拉全部 89 个 task 镜像；起或复用 SGLang 的 OpenAI 兼容端点；生成 Harbor config 并强制自检代理变量、`DEBIAN_FRONTEND=noninteractive`、`TZ=Etc/UTC`、`model_info` 四项；在 tmux 里启动全量评测；轮询 job result、容器列表、SGLang 健康直到 `finished_at` 非空。这套流程在 2026-07-01 至 07-02 完整跑通过一次：89/89 全部落盘，Harbor 聚合分数 2.0/89 = 2.25%，20 个 task 级 `AgentTimeoutError`。最关键的一条判据是：**task 级超时是评测结果，不是基础设施故障，不要因此介入**；只有 SGLang 退出、Docker 不通、Harbor 主进程退出但 job 未写 `finished_at` 才需要人工处理。本手册用 `terminus-2` agent；要跑训练侧对齐的 camel-agent（mode B）评测，把第六步换成 [`HARBOR_CAMEL_MODE_B_zh.md`](HARBOR_CAMEL_MODE_B_zh.md) 里的 launcher，其余六步完全通用。

## 1. 为什么必须这么配

这台 worker 不是裸机，下面每一行的"必须配置"都是踩过对应失败模式之后固定下来的，不是风格选择。

| 层 | 环境约束 | 必须配置 | 不这样做的失败模式 |
|---|---|---|---|
| Docker daemon | pod 内普通嵌套 dockerd 受 cgroup / capability 限制 | rootless Docker + fuse-overlayfs + slirp4netns | `mkdir /sys/fs/cgroup/...: read-only file system`，`docker run` 失败 |
| Docker data root | 共享存储对 layer unpack 的 chown 不友好 | data root 放 `/tmp/tbv21-rootless-docker-$USER` | unpack 时 `Lchown ...: operation not permitted` |
| Docker API | dockerd 26.1.3 在该 rootless 组合下 `/version` 协商可能 EOF | 固定 `DOCKER_API_VERSION=1.45`，健康判据用 `docker info` 而非 `docker version` | 把可用的 daemon 误判为坏的 |
| Docker pull | 到 Docker Hub 的直连可能 TLS 超时 | daemon 配 registry mirror + 评测前预拉 + 镜像站 fallback | 全量评测中途卡在拉镜像 |
| 容器网络 | task 容器内要访问 apt / pypi / 外部数据 | Docker CLI 代理 + Harbor env 代理 + `APT_CONFIG` | apt / pip / 下载超时 |
| apt 交互 | tzdata 等包会等待交互输入 | `DEBIAN_FRONTEND=noninteractive` + `TZ=Etc/UTC` | task 长时间不动，看起来像 Harbor 卡死 |
| 模型配置 | LiteLLM 不认识本地 served model | Harbor agent config 里填 `model_info` 的 token 上限与 cost | unknown model / context fallback |
| SGLang 管理 | pid 文件可能缺失但服务健康 | 先探 `/v1/models`，健康就复用 | 误杀正在工作的模型服务 |
| task 运行时 | 镜像内不一定有合适的 tmux / uv | bind mount 预置的 tmux、tools、wheelhouse | agent terminal session 初始化失败、依赖装不上 |

这条路线和 CPU worker 上的 `docker-env-server` 不是一回事：后者是 GPU worker 通过 HTTP 访问远端 pool_server，用于 RL rollout；本手册是 Harbor 直接用 Docker backend 在本机起 task container。不要把 `WORKER_URLS=http://<cpu>:18081` 当成 Harbor 的 Docker backend，Harbor 需要的是能 `docker run` / `docker compose` / 建网络 / 建 bind mount 的 Docker API。

## 2. 网络资源的四层处理

TBv2.1 的任务会在运行中访问公网或内网镜像源，处理方式是四层叠加，而不是逐个任务打补丁。

第一层是镜像预拉。评测前用 `bin/prepull_tbv21_images.sh` 从 89 个 `task.toml` 解析 `docker_image` 并逐个 `docker pull`，直连失败时改拉 `<mirror>/<原始 image>` 再 `docker tag` 回原名。必须看到 `total=89 ok=89 failed=0` 才能进入下一步。

第二层是 rootless daemon 自身的代理与 registry mirror，写在 `~/.config/docker/daemon.json` 的 `registry-mirrors` 和 `proxies` 字段，解决 daemon 拉镜像时的网络问题。

第三层是 Docker CLI 与 task container 的环境变量。`env.sh` 会按当前 shell 的代理重写 `docker-cli/config.json` 的 `proxies.default`，Harbor config 则给容器注入 `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` 及其小写形式、`DEBIAN_FRONTEND=noninteractive`、`TZ=Etc/UTC`、`APT_CONFIG=/opt/tbv21-apt/apt.conf`。

第四层是 apt 与 Python 依赖的专项配置。apt 不一定读普通环境变量，所以显式生成 `runtime/apt/apt.conf` 并 bind mount 进容器，内容是 `Acquire::http::Proxy` / `Acquire::https::Proxy` / 重试与超时。Python 侧通过 `UV_INDEX_URL` / `UV_EXTRA_INDEX_URL` / `UV_TRUSTED_HOST` 指向内网 PyPI 镜像，并用 `UV_FIND_LINKS=/opt/tbv21-wheelhouse` 提供 pytest 生态的离线 wheel。wheelhouse 不是全量离线库，其余依赖仍然依赖镜像源或代理可达。

这四层能解决基础设施层面的网络问题，但解决不了任务自身要下载的外部数据慢或失效。这类情况表现为 task 级超时，应记为评测结果。`caffe-cifar-10` 就是典型：加了 `DEBIAN_FRONTEND=noninteractive` 之后不再卡 tzdata 交互，但仍可能因 CIFAR 下载或任务本身耗时触发 `AgentTimeoutError`。

## 3. 环境变量与目录约定

评测 bundle 的根目录由 `TBV21_HOME` 指定，本手册验证时使用的是集群上的 `/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1`。`source ./env.sh` 负责设置路径、代理和 Docker CLI 配置，但它不保证 Docker daemon 已启动。顺序规则是硬性的：先 `bash bin/use_worker_rootless_docker.sh start` 起 daemon，再 `source "$TBV21_DOCKER_HOST_ENV"` 让当前 shell 的 `DOCKER_HOST` 与实际 daemon 对齐，之后才能执行任何 `docker info` / `docker pull` / `harbor run`。

代理相关的变量优先级是：当前 shell 已有的 `HTTP_PROXY` 优先于 `TBV21_PROXY_URL`，后者再优先于内置默认值。换 worker 或换代理时显式设置后重新 source：

```bash
export TBV21_PROXY_URL=http://<proxy-host>:<port>
export HTTP_PROXY="$TBV21_PROXY_URL"
export HTTPS_PROXY="$TBV21_PROXY_URL"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export NO_PROXY='localhost,127.0.0.1,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,.pjlab.local,.svc'
export no_proxy="$NO_PROXY"
source ./env.sh
```

必须让本地 SGLang 绕过代理，所有探活都要带 `--noproxy '*'`：`curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:30000/v1/models`。旧代理值如果被锁在手写的 `docker-cli/config.json` 里，会导致容器内 apt / pip 卡住，所以默认让 `env.sh` 重写该文件；确有需要保留手写配置时才设 `TBV21_DOCKER_CONFIG_REWRITE=0`。

## 4. 新 worker 上线检查清单

按顺序执行，不要跳步。

第一步核对 bundle 资源：89 个 task 目录、`bin/harbor`、`bin/python_sglang`、`runtime/tmux_runtime/bin/tmux`、`runtime/tools/bin/uv`、`runtime/wheelhouse`、`docker-cli-plugins/docker-compose` 全部存在且可执行，模型 symlink 的目标 `config.json` 可读。`bin/harbor` 和 `bin/python_sglang` 是指向共享 conda 环境的 wrapper，必须验证 wrapper 目标真的可执行（`bin/harbor --version`），目标不可读时先修 Python 环境，不要继续。

第二步核对集群能力：当前用户有 NOPASSWD sudo（rootless Docker wrapper 依赖它），`docker` 与 `dockerd` 二进制存在，宿主有 `tmux`，代理可达。sudo 不可用时这套 wrapper 无法按当前方式自动启动，需要改用集群支持的其他路线。

第三步起 rootless Docker。`/tmp` 至少预留 60 GiB。首次需要先跑 `bin/setup_worker_rootless_docker.sh` 安装 rootless extras 与 slirp4netns。启动后确认 `docker info` 显示 driver 是 `fuse-overlayfs`、root 在 `/tmp/tbv21-rootless-docker-<user>` 而不是共享存储。

第四步跑容器联网 smoke（`bin/use_worker_rootless_docker.sh smoke`），必须看到 hello-world 与 alpine 访问外网都成功，否则不要进入 Harbor。

第五步预拉镜像，必须 `failed=0`。第六步起 SGLang 并确认 `/v1/models` 返回目标模型。第七步跑单任务 smoke，能建容器、能调模型、能写出 `result.json` 之后，再跑全量。

## 5. 全量评测与监控

在专用 tmux session 里启动全量评测，不要依赖某个约定的 window 编号存在：

```bash
TARGET_PANE=tbv21_full_eval:0.0
tmux has-session -t "${TARGET_PANE%%:*}" 2>/dev/null || tmux new-session -d -s "${TARGET_PANE%%:*}" -n eval
tmux send-keys -t "$TARGET_PANE" "cd '$TBV21_HOME' && source ./env.sh" C-m
tmux send-keys -t "$TARGET_PANE" "export TBV21_FULL_EVAL_JOB_NAME=full_eval_tbv21_${TBV21_MODEL_NAME}_\$(date +%Y%m%d_%H%M%S)" C-m
tmux send-keys -t "$TARGET_PANE" "export TBV21_GPU_IDS=0,1,2,3 TBV21_FULL_EVAL_CONCURRENCY=2 TBV21_FULL_EVAL_MAX_RETRIES=1" C-m
tmux send-keys -t "$TARGET_PANE" "bash bin/run_full_eval_qwen3_8b.sh" C-m
```

启动后 10 秒内必须确认 `logs/${JOB}.log` 已生成且非空，否则说明 tmux target 写错了，实际没跑起来。脚本名里的 `qwen3_8b` 是历史命名，它读的是 `TBV21_MODEL_NAME` 和 `TBV21_MODEL_PATH`，换模型不需要改脚本名。脚本在调用 Harbor 之前会自检 config，必须看到 `[OK] Harbor config has proxy env, noninteractive apt env, and model_info`；自检失败时修 config 生成逻辑，不要手工绕过去直接起 Harbor。

监控每 5 分钟轮询一次，同时看四样东西：`jobs/$JOB/result.json` 的 `finished_at` 与已落盘 trial 数、`docker ps` 的活跃容器、Harbor 主日志尾部、SGLang `/v1/models` 探活。正常推进的表现是已完成数持续增加，或当前容器的运行时长还没超过该 task 的 timeout；`docker ps` 里有一两个 task 容器且名字随任务完成不断更换；`AgentTimeoutError` 计数增加但新任务继续启动。

判断是否需要介入只看基础设施：SGLang 探活失败且日志显示进程退出、`docker info` 失败或 daemon 不通、`docker ps` 卡死、Harbor 主进程退出但 job 没有 `finished_at`、某容器远超 task timeout 仍未释放且 Harbor 没写该 task 的结果。以下情况不要介入：单个 task 跑 15 到 60 分钟（TBv2.1 有任务 timeout 为 3600 秒，个别更久）、`AgentTimeoutError`、主日志里的 `Unclosed client session` 警告。

## 6. 结果口径

Harbor 的聚合分数分母是 `n_total_trials`，报告时统一用这个口径。验证运行的数字是 `reward_sum = 2.0`、`n_total_trials = 89`，即 `2.0 / 89 = 0.0224719101`；如果只对有 `reward` 字段的 88 个结果求均值会得到 `2.0 / 88 = 0.0227272727`，这不是报告口径。`error_counts` 里的 `AgentTimeoutError: 20` 是 task / 模型层面的超时，不是 Harbor、Docker 或 SGLang 崩溃。

## 7. 验证运行的完整记录

| 项 | 值 |
|---|---|
| 模型 | `qwen3-8b-rl-iter215` |
| SGLang | `http://127.0.0.1:30000/v1`，TP=4 |
| Docker | rootless，driver `fuse-overlayfs`，root `/tmp/tbv21-rootless-docker-<user>`，ServerVersion 26.1.3 |
| Job | `full_eval_tbv21_qwen3-8b-rl-iter215_ready_20260701_215702` |
| 起止 | 2026-07-01T22:00:13 至 2026-07-02T07:22:00 |
| 完成度 | 89 / 89 |
| 聚合分数 | 2.0 / 89 = 0.0224719101 |
| 错误分布 | `AgentTimeoutError: 20` |
| 解出任务 | `configure-git-webserver__zVAaSVv`、`hf-model-inference__72rnD2H` |
| 收尾状态 | 无残留 task container，SGLang 保持健康 |

job 名里的 `_ready_20260701_215702` 只是这次验证运行的历史标记，不是脚本要求；新的运行用 `full_eval_tbv21_${TBV21_MODEL_NAME}_$(date +%Y%m%d_%H%M%S)` 即可。

## 8. 常见问题

镜像拉取报 TLS handshake timeout 或 context deadline exceeded 时，跑预拉脚本并确认镜像站 fallback 生效。容器内 apt 卡住或 tzdata 等待交互时，检查 `DEBIAN_FRONTEND`、`TZ`、`APT_CONFIG` 和代理四项是否都注入了。LiteLLM 报 unknown model 或 context fallback 时，检查 Harbor agent config 是否带了 `model_info` 的 `max_input_tokens` / `max_output_tokens`，只写 `model_name` 不够。SGLang pid 文件缺失时先用 `/v1/models` 探活，健康就复用，不要杀 tmux session。Harbor 跑完仍有容器残留时，先确认 job 已写 `finished_at`，再按名字删对应容器，不要 `docker prune` 镜像。rootless daemon 日志里的 cgroup 清理噪声可以忽略，判据是 `docker ps -a` 无残留、Harbor result 完成、SGLang 健康三项。

## 9. Ready 判据

同时满足以下六条才算这套流程就绪：所有脚本 `bash -n` 通过；`docker info` 显示 rootless + fuse-overlayfs 且 root 在大容量盘；预拉输出 `failed=0`；SGLang `/v1/models` 正常返回目标模型；Harbor config 自检输出 `[OK]`；`jobs/$JOB/result.json` 最终有非空 `finished_at` 且 `docker ps -a` 无残留 task container。
