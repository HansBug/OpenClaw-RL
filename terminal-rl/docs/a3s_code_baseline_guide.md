## 结论

新增脚本 `terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh` 是 a3s-code + DAPO SetA baseline wrapper，默认只覆盖 harness/data/algo/DAPO 关键变量，其余训练逻辑复用 `terminal-rl_qwen3-8b_pu.sh`，减少脚本漂移。

## 使用方式

```bash
# 只检查最终 train_async 命令
DRY_RUN=1 bash terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh

# 正式运行示例
WORKER_URLS=http://<worker-ip>:18081 \
bash terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh
```

## 关键默认值

| 变量 | 默认 | 位置 / 说明 |
|---|---|---|
| `HARNESS_OPTION` | `a3s-code` | `terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh:16` |
| `DATASET` | `seta` | `terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh:17` |
| `ALGO` | `dapo` | `terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh:18` |
| `SETA_SAFETY` | `clawsentry` | SetA + safety reward baseline |
| `SAFETY_REWARD_COEF` | `0.3` | 与现有 SetA safety baseline 对齐 |
| `MAX_TURN` | `10` | 与现有 DAPO SetA run tag 对齐 |
| `DAPO_EPS_CLIP_LOW/HIGH` | `0.2 / 0.28` | DAPO clipping |
| `DAPO_CALCULATE_PER_TOKEN_LOSS` | `1` | token-level loss |
| `DAPO_DYNAMIC_SAMPLING` | `1` | dynamic sampling |
| `A3S_CODE_REPO_ROOT` | `/mnt/shared-storage-user/puyuan/code/a3s-lab/Code` | `terminal-rl/terminal-rl_qwen3-8b_pu.sh:274` |
| `A3S_CODE_CONFIG_PATH` | `${REPO_ROOT}/a3s-code-adapter/generated_configs/a3s-code-shared.hcl` | `terminal-rl/terminal-rl_qwen3-8b_pu.sh:275` |

## Pipeline 兼容性

| 模块 | 兼容性结论 |
|---|---|
| `router_server` | SetA 仍会 reset/evaluate terminal env；a3s SDK turn 本身不走 terminal-rl 外部 tool-call loop |
| ClawSentry | reward-level 流程保留；pre-action 只覆盖 terminal-rl 外部 `tool_call_requests`，SDK 内部工具调用当前只审计 |
| PRM | PRM 仍可按模型 turn 记录 assistant output；a3s 内部工具细节在 `sdk_tool_calls`，未映射为外部 tool result |
| trajectory 保存 | `generate.py:1594` 写入 `harness_option`、`sdk_model_turns`、`sdk_tool_calls`，旧 `tool_calls` 字段保留 |
| lease 回收 | `generate.py:2091` 先 `agent_runner.close()`，再 `env_client.close(lease_id)`，释放顺序明确 |

## 推荐命令

```bash
# a3s-code DAPO SetA baseline
HARNESS_OPTION=a3s-code \
DATASET=seta \
ALGO=dapo \
WORKER_URLS=http://<worker-ip>:18081 \
bash terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh

# 覆盖 SDK 配置路径
A3S_CODE_REPO_ROOT=/path/to/a3s-lab/Code \
A3S_CODE_CONFIG_PATH=/path/to/a3s-code-shared.hcl \
bash terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh
```

## 运行前检查

| 检查项 | 命令 |
|---|---|
| wrapper dry-run | `DRY_RUN=1 bash terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh` |
| SDK import | `${TRAIN_PYTHON:-python3} -c "import a3s_code"` |
| config exists | `test -f "$A3S_CODE_CONFIG_PATH"` |
| pool server | `curl http://<worker-ip>:18081/healthz` |
