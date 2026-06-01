## 结论

两条 harness 的配置路由、构造、turn 执行、trajectory 记录和 close 降级已做兼容修复；本地 mock/DRY_RUN 均通过。真实 PR commit `fa1f857` 未能 fetch，真实 GPU/Ray/a3s SDK 端到端未跑，是当前主要风险。

## A-E 兼容性检查

| 维度 | 结论 | 修复内容 | 修复 commit |
|---|---|---|---|
| A. `agent_runner.py` | 兼容 | `normalize_harness_option()` 覆盖下划线/连字符；camel 路径不依赖 `env_client/lease_id`；自定义 `run_model_turn` 优先，缺失时回退旧 SGLang；`close()` 安全 no-op | `3f204a2fa6e2` |
| B. `custom_types.py` | 兼容 | `TurnResult.interactions` 为 Optional；旧 camel 构造仍只需 `interaction`；`generate.py` 用兜底读取 | `3f204a2fa6e2` |
| C. `generate.py` | 兼容 | agent 类型优先级为 `harness_option > terminal_agent_type > camel_agent`；新增 `sdk_model_turns/sdk_tool_calls` 为 additive 字段；`agent_runner.close()` 在 env lease close 前 best-effort 执行 | `3f204a2fa6e2` |
| D. rollout yaml | 兼容 | `rollout_qwen3.yaml:3`、`rollout_qwen3_think.yaml:3` 默认 `harness_option: camel-agent` | `3f204a2fa6e2` |
| E. `terminal-rl_qwen3-8b_pu.sh` | 兼容 | 默认 `HARNESS_OPTION=camel-agent`；a3s SDK 安装只在 `a3s-code && DRY_RUN!=1`；A3S runtime env 仅 a3s 注入；Ray job 用 `${TRAIN_PYTHON}` | `3f204a2fa6e2` |

## 自测结果

| 命令 | 结果 | 关键输出 |
|---|---|---|
| `python -m pytest ...` | 系统 Python 不可用 | 初始环境缺 pytest；改用 `/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python` |
| `.../python -m pytest terminal-rl/tests/test_a3s_code_agent.py -v` | 通过 | `1 passed in 5.10s` |
| `.../python -m pytest terminal-rl/tests/test_agent_runner_harness_option.py -v` | 通过 | `4 passed in 4.94s` |
| `.../python -m pytest terminal-rl/tests/test_harness_option_routing.py -v` | 通过 | `2 passed in 0.39s` |
| `HARNESS_OPTION=camel-agent DRY_RUN=1 bash terminal-rl/terminal-rl_qwen3-8b_pu.sh` | 通过 | `Harness:  camel-agent`，打印 `[dry-run] python3 -u ... train_async.py` |
| `HARNESS_OPTION=a3s-code DRY_RUN=1 bash terminal-rl/terminal-rl_qwen3-8b_pu.sh` | 通过 | `Harness:  a3s-code`，打印最终 `train_async.py` 命令 |
| `DRY_RUN=1 bash terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh` | 通过 | `ALGO=dapo`，`Harness:  a3s-code`，DAPO 参数进入最终命令 |

## 已知风险 / TODO

| # | 风险 / TODO | 状态 |
|---|---|---|
| 1 | 网络恢复后需 fetch PR #8，对 `fa1f857` 做逐文件 parity review | 未完成 |
| 2 | 当前 a3s SDK 只用 mock 测试；未跑真实 SDK + Ray + GPU 端到端 | 未完成 |
| 3 | a3s SDK 内部工具调用目前只进入 `sdk_tool_calls` 审计字段；terminal-rl 的 ClawSentry pre-action/PRM 工具级记录只覆盖外部 `tool_call_requests` | 已标注 |
| 4 | 真实运行前必须确认 `A3S_CODE_REPO_ROOT` 与 `A3S_CODE_CONFIG_PATH` 存在且 SDK 可 import | 已标注 |
