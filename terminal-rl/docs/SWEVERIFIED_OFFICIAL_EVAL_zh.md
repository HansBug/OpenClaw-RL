# SWE-bench Verified 官方评测适配与使用说明

## 1. 范围与结论

本适配在现有 SWE-smith `Docker worker` 基础上增加
SWE-bench Verified 的标准评测链路，目标模型为 `Qwen/Qwen3-8B` 基模，
默认开启 thinking mode。

SWE-bench Verified 是 **eval benchmark**，不是训练数据集。完整流程分为：

1. 固定版本的 500 条 Verified 数据转换与任务目录生成；
2. `terminal-rl` + Qwen3-8B 在隔离的 Docker workspace 中生成代码修改；
3. worker 导出官方 prediction schema 所需的 `model_patch`；
4. 使用固定版本的官方 `swebench.harness.run_evaluation` 计算最终分数。

worker 内部不会自行宣称 `resolved`，也不会用自定义测试结果替代官方分数。
只有第 4 步输出的官方 harness report 才是最终评测结果。

固定版本：

- dataset：`princeton-nlp/SWE-bench_Verified`
- split：`test`
- dataset revision：`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`
- instance 数：`500`
- SWE-bench：`4.1.0`
- harness commit：`f7bbbb2ccdf479001d6467c9e34af59e44a840f9`

官方参考：

- [SWE-bench evaluation guide](https://www.swebench.com/SWE-bench/guides/evaluation/)
- [SWE-bench harness reference](https://www.swebench.com/SWE-bench/reference/harness/)
- [SWE-bench Verified dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)

## 2. 关键文件

| 文件 | 用途 |
|---|---|
| `terminal-rl/data_utils/convert_sweverified_to_terminal_rl.py` | 固定版本数据转换、官方 image 映射、任务目录生成 |
| `terminal-rl/data_utils/download_sweverified.sh` | `smoke/full` 数据准备入口和 artifact lock |
| `terminal-rl/remote/run_pool_server_sweverified_pu.sh` | 端口 `18083` 的独立 worker 启动与 500 条只读预检 |
| `terminal-rl/scripts/smoke_swe_worker.py` | SWE-smith / SWE-Verified 通用 worker smoke client |
| `terminal-rl/terminal-rl_qwen3-8b_eval_pu.sh` | 通用 Qwen3-8B eval-only launcher |
| `terminal-rl/scripts/run_sweverified_qwen3_8b_base_think_eval.sh` | 4 卡 Qwen3-8B Verified 正式评测入口 |
| `terminal-rl/swebench_report.py` | 导出 `predictions.jsonl` 与 generation coverage，不计算官方分数 |
| `terminal-rl/scripts/run_swebench_verified_official_harness.sh` | 固定官方 commit 的最终评分入口 |

## 3. 数据准备

在共享文件系统中的 repo 根目录执行：

```bash
cd /path/to/OpenClaw-RL

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
source <(curl -fsSL \
  http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh)

MODE=full \
PYTHON_BIN=/root/miniconda3/bin/python3 \
bash terminal-rl/data_utils/download_sweverified.sh
```

正式转换强制从固定 Hugging Face revision 读取完整 500 条数据，不允许
`INPUT_JSONL` 或行数截断。预期输出：

```text
terminal-rl/dataset/sweverified_convert/test.jsonl
terminal-rl/dataset/sweverified_convert/convert_stats.json
terminal-rl/dataset/sweverified_env/<instance_id>/
```

`test.jsonl` 的预期 SHA256：

```text
4282529dbcc1b9253fa91da35b9f1768a2002b391cc90ac6a4e64575d59cfbf3
```

需要重建时显式增加 `OVERWRITE=1`。转换和 worker 共享
`.sweverified_artifact.lock`，避免 worker 读取发布中的 artifact。

## 4. 启动 Docker Worker

在有 Docker 的开发机执行：

```bash
cd /path/to/OpenClaw-RL

ENV_SERVER_PORT=18083 \
WORKER_MAX_TASKS=4 \
WORKER_MAX_RUNS_PER_TASK=2 \
WORKER_MAX_CONCURRENT_BUILDS=1 \
WORKER_MAX_CONCURRENT_RESETS=4 \
WORKER_MAX_CONCURRENT_CLOSES=8 \
WORKER_MIN_DOCKER_FREE_GB=120 \
CONTAINER_MEMORY_LIMIT=16g \
CONTAINER_PIDS_LIMIT=256 \
bash terminal-rl/remote/run_pool_server_sweverified_pu.sh
```

该服务使用独立的 `TERMINAL_RL_POOL_NAMESPACE=sweverified` 和端口
`18083`。默认关闭 broad cleanup；SETA、SWE-smith 和 SWE-Verified 的
Compose start/down、watchdog network cleanup 通过同一个 host file lock
串行化，可与端口 `18081/18082` 的服务共存。

仅检查完整 500 条 artifact 而不启动服务：

```bash
WORKER_PREFLIGHT_ONLY=1 \
POOL_SERVER_PYTHON=/root/miniconda3/bin/python3 \
bash terminal-rl/remote/run_pool_server_sweverified_pu.sh
```

## 5. Worker Smoke

从 GPU 节点执行：

```bash
cd /path/to/OpenClaw-RL

python3 terminal-rl/scripts/smoke_swe_worker.py \
  --suite sweverified \
  --worker-url http://<docker-worker-host>:18083 \
  --index 0 \
  --ensure-image-timeout 3600 \
  --reset-session-timeout 900
```

成功条件包括：

- `/healthz`、`/allocate`、`/reset`、`/exec_tool`、`/evaluate`、`/close`
  全链路成功；
- agent 工作目录固定为 `/testbed`；
- `/evaluate` 返回 `grader=swebench_prediction_export`；
- 导出的 `model_patch` 包含 smoke probe；
- worker 不运行本地自定义 grader。

## 6. 4 卡 Qwen3-8B 正式生成

在 4 卡 H20 GPU 节点执行：

```bash
cd /path/to/OpenClaw-RL

WORKER_URLS=http://<docker-worker-host>:18083 \
HF_CKPT=/path/to/Qwen3-8B \
REF_LOAD=/path/to/Qwen3-8B_torch_dist \
WANDB_MODE=offline \
bash terminal-rl/scripts/run_sweverified_qwen3_8b_base_think_eval.sh
```

固定正式配置：

- 500 instances，`n_samples=1`；
- 4 张 rollout GPU；
- 2 个 `TP=2` SGLang engine；
- `EVAL_MAX_CONCURRENCY=4`；
- thinking mode；
- `temperature=0.6`、`top_p=0.95`、`top_k=20`；
- worker 仅导出 patch，不在生成阶段计算 `resolved`。

生成阶段只有同时满足以下条件才返回成功：

- `submitted=500`；
- `incomplete=0`；
- `unexpected=0`；
- `technical_failures=0`；
- 500 个 instance 全部处于 `pending_official_grading`。

关键输出位于：

```text
runs/<run_id>/swebench_official/predictions.jsonl
runs/<run_id>/swebench_official/prediction_coverage.json
runs/<run_id>/swebench_official/instance_audit.json
runs/<run_id>/swebench_official/score_summary.json
```

其中 `score_summary.json` 的 `authoritative_score` 必须为 `null`，用于明确
表示官方评分尚未执行。

## 7. 官方 Harness 最终评分

在有 Docker 且能拉取官方 SWE-bench image 的机器执行：

```bash
cd /path/to/OpenClaw-RL

RUN_DIR=/path/to/OpenClaw-RL/runs/<run_id> \
MAX_WORKERS=4 \
EVAL_TIMEOUT=1800 \
bash terminal-rl/scripts/run_swebench_verified_official_harness.sh
```

脚本会：

1. checkout 固定 commit，并以 editable mode 安装 `swebench==4.1.0`；
2. 校验安装来源 commit；
3. 校验 `predictions.jsonl` 恰好包含 500 条、ID 唯一、字段严格为
   `instance_id/model_name_or_path/model_patch`；
4. 在启动 Docker 前 import 完整 harness，防止普通 wheel 漏打包
   `constants/fixtures`；
5. 调用官方 `swebench.harness.run_evaluation`。

最终结果位于：

```text
runs/<run_id>/swebench_official/harness/
```

以该目录中的官方 report 统计 `resolved` 与 `resolved rate`。生成阶段的
`reward=0` 只是 deferred grading 占位值，不能当作模型得分。

## 8. 已有本地评测审计

历史 run：

```text
runs/eval_qwen3-8b_prerl_sweverified_fixed_2026-07-22_234611
```

截至 2026-07-23 12:37：

- progress 停在 `251/500`；
- 已落盘 `248` 个 trajectory；
- status：`197 truncated / 49 completed / 2 failed`；
- router 中有 3 次 `/reset HTTP 500`，后续请求可继续，属于 worker
  生命周期稳定性问题；
- 没有完整 500 条 `predictions.jsonl`；
- 没有官方 harness aggregate report；
- 当前没有对应 SGLang/RolloutManager 进程。

因此该 run **未完成 SWE-bench Verified 官方评测**。它只证明旧链路能够
长时间运行并处理约半数样本，不能作为最终 benchmark 结果。

本分支已完成的本地验证：

- 固定官方 Hugging Face revision 完整转换 `500/500`；
- 转换后任务目录 `500/500`；
- 数据 SHA256 与 launcher 固定值一致；
- 500 个 task 内容指纹逐条通过 worker read-only preflight；
- Qwen3-8B 固定 revision、4 卡 `2 x TP=2` topology 的正式 launcher
  `EVAL_DRY_RUN=1` 通过；
- 官方 harness `4.1.0@f7bbbb2...` 安装来源、500 条 prediction schema
  与 ID 唯一性 preflight 通过；
- SWE-Verified + SWE-smith 相关回归测试 `72 passed`；
- 除仓库已有 `router_server_readyz` 测试桩不兼容外，其余全仓测试
  `161 passed`；该 2 项失败位于未修改的 router/test 文件，原因是旧
  `_FakeRouter` 缺少现有 `maybe_reload_workers()` 方法；
- Python compile、shell `bash -n`、`git diff --check` 通过。

完整任务的验收标准是：

1. generation 阶段导出完整 500 条 predictions 且无技术失败；
2. 固定官方 harness 完成 500 条评分；
3. 保存官方 aggregate report 和 instance logs。
