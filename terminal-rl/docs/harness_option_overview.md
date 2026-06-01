## 结论

`dev-agenticrl-safety-exploration-harness` 已加入 `camel-agent` / `a3s-code` 双 harness 路由，默认仍是 `camel-agent`。PR #8 的远端 commit `fa1f857` 在当前环境不可 fetch，`git show fa1f857` 也显示本地不存在；本分支按 PR 描述完成手工兼容实现，后续需在网络可用时做一次逐文件 parity review。

## 合入摘要

| 项 | 结果 |
|---|---|
| base | `dev-agenticrl-safety-exploration` at `fc872aa8` |
| work branch | `dev-agenticrl-safety-exploration-harness` |
| harness commit | `3f204a2fa6e2` |
| PR commit | `fa1f857` 未在本地仓库，GitHub fetch/pull 受代理 403 限制 |
| default behavior | `rollout_qwen3*.yaml` 和 `pu.sh` 默认 `camel-agent` |

## Diff Stat

```text
 terminal-rl/agent/a3s_code_agent.py                | 358 +++++++++++++++++++++
 terminal-rl/agent_runner.py                        |  66 +++-
 terminal-rl/configs/rollout_qwen3.yaml             |   1 +
 terminal-rl/configs/rollout_qwen3_think.yaml       |   1 +
 terminal-rl/custom_types.py                        |   1 +
 terminal-rl/generate.py                            |  44 ++-
 terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh         |  30 ++
 terminal-rl/terminal-rl_qwen3-8b_pu.sh             |  76 ++++-
 terminal-rl/tests/test_a3s_code_agent.py           | 111 +++++++
 terminal-rl/tests/test_agent_runner_harness_option.py | 148 +++++++++
 terminal-rl/tests/test_harness_option_routing.py   |  24 ++
 11 files changed, 847 insertions(+), 13 deletions(-)
```

## Harness 路由总览

| 阶段 | camel-agent | a3s-code |
|---|---|---|
| 配置入口 | `terminal-rl/configs/rollout_qwen3.yaml:3` | `HARNESS_OPTION=a3s-code` 或 `terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh:16` |
| shell 默认 | `terminal-rl/terminal-rl_qwen3-8b_pu.sh:122` | wrapper 覆盖后复用 `pu.sh` |
| 参数归一化 | `terminal-rl/agent_runner.py:31` 支持 `camel_agent` / `camel-agent` | 同处支持 `a3s_code` / `a3s-code` |
| 训练入口 | `terminal-rl/generate.py:1452` 读取 `harness_option` | 同入口，优先级 `harness_option > terminal_agent_type > camel_agent` |
| Agent 创建 | `terminal-rl/agent_runner.py:167` 创建 `CamelAgent` | `terminal-rl/agent_runner.py:176` 创建 `A3SCodeAgent` |
| 模型 turn | 无 `run_model_turn` 时走旧 SGLang fallback：`terminal-rl/agent_runner.py:120` | 优先 agent 自己的 `run_model_turn`：`terminal-rl/agent_runner.py:104` |
| 工具调用 | terminal-rl 执行 `env_client.exec_tool`，保留旧逻辑 | SDK 在 `session.send()` 内执行工具，terminal-rl 记录 `sdk_tool_calls` |
| 收尾 | `AgentRunner.close()` 对无 `close` 的 camel-agent no-op：`terminal-rl/agent_runner.py:144` | `A3SCodeAgent.close()` 释放 SDK session：`terminal-rl/agent/a3s_code_agent.py:349` |

## 关键路径

| 文件 | 要点 |
|---|---|
| `terminal-rl/custom_types.py:78` | `TurnResult.interactions` 为 Optional，旧调用不需要改 |
| `terminal-rl/generate.py:1582` | `interactions` 新旧兜底：`turn_state.interactions or [turn_state.interaction]` |
| `terminal-rl/generate.py:1594` | `turn_records` 新增 `harness_option` / `sdk_model_turns` / `sdk_tool_calls` |
| `terminal-rl/terminal-rl_qwen3-8b_pu.sh:327` | 仅非 dry-run 且 `a3s-code` 时尝试安装 SDK |
| `terminal-rl/terminal-rl_qwen3-8b_pu.sh:1173` | runtime `PYTHONPATH` 只在 `a3s-code` 下追加 adapter |
