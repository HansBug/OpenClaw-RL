# AgentHarm / AgentSafetyBench 官方安全评测使用说明

本文档说明如何在 Terminal-RL 中对已完成的 eval run 生成 `AgentHarm` 与 `AgentSafetyBench` 官方口径 split 指标。整体流程分两段：

1. 先用 Terminal-RL 生成每个 checkpoint 的 eval trajectory。
2. 再从 trajectory 中导出/调用官方 scorer，并汇总成最终得分表。

`AgentHarm` 指标直接来自 trajectory 中保留的官方 scorer 语义字段；`AgentSafetyBench` 官方真实分数必须实际运行官方 `ShieldAgent` judge。

## 1. 整体 Eval 流程

推荐流程：

| 阶段 | 输入 | 输出 | 主要脚本 |
| --- | --- | --- | --- |
| 生成评估轨迹 | checkpoint + eval suite | `runs/eval/<run>/trajectories/*/meta.json` | Terminal-RL eval 启动脚本 |
| 准备 ASB judge 输入 | ASB trajectories | `runs/official_asb_shield_inputs/<target>/gen_res.json` | `terminal-rl/scripts/prepare_asb_shield_inputs.py` |
| 运行官方 ASB judge | `gen_res.json` + `ShieldAgent` | `<Agent-SafetyBench>/score/shield_results/<target>/` | `terminal-rl/scripts/run_safety_official_eval.sh` |
| 汇总最终得分 | trajectories + ASB judge 输出 | markdown score table | `terminal-rl/scripts/summarize_safety_eval.py` |

常用入口是：

```bash
bash terminal-rl/scripts/run_safety_official_eval.sh <name>=<eval_run_dir>
```

该 wrapper 会自动完成 ASB 输入导出、ShieldAgent 调用和最终 summary。只想单独准备数据或单独汇总时，可以直接调用 Python 工具。

## 2. 数据集简介

### AgentHarm

`AgentHarm` 评估 LLM agent 在 tool-use 场景中面对有害任务时的拒答能力，以及面对无害任务时的正常完成能力。

| 项 | 信息 |
| --- | --- |
| paper | `AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents` |
| arXiv | `https://arxiv.org/abs/2410.09024` |
| HuggingFace | `https://huggingface.co/datasets/ai-safety-institute/AgentHarm` |
| Inspect Evals 页面 | `https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/agentharm/` |
| 本地官方实现 | `/mnt/shared-storage-user/puyuan/code/inspect_evals/src/inspect_evals/agentharm` |

本仓当前使用 validation split：

| split | 本地文件 | 样本数 | 正确行为 |
| --- | --- | ---: | --- |
| harmful | `terminal-rl/dataset/agentharm/harmful_validation.jsonl` | 32 | 拒绝或避免协助有害目标 |
| benign | `terminal-rl/dataset/agentharm/harmless_benign_validation.jsonl` | 32 | 完成无害任务，不应拒答 |

官方指标：

| 指标 | 计算方式 |
| --- | --- |
| `AH official harmful refusal rate` | harmful split 中 `reward_details.refused=True` 的比例 |
| `AH official benign full-score accuracy` | benign split 中 `reward_details.score == 1.0` 的比例，对应 inspect-evals `avg_full_score` |

注意：用于正式 AH 汇报的 eval run 必须确保 `reward_details.score` 保留官方 scorer 语义，不能用训练用 shaped reward 替代。

### AgentSafetyBench

`AgentSafetyBench` 评估 agent 环境中的工具调用安全性，覆盖网页、文件、数据库、邮件、搜索等风险场景。官方使用 `ShieldAgent` 对完整行为轨迹进行 safe/unsafe 判断。

| 项 | 信息 |
| --- | --- |
| paper | `Agent-SafetyBench: Evaluating the Safety of LLM Agents` |
| arXiv | `https://arxiv.org/abs/2412.14470` |
| GitHub | `https://github.com/thu-coai/Agent-SafetyBench` |
| HuggingFace data | `https://huggingface.co/datasets/thu-coai/Agent-SafetyBench` |
| ShieldAgent model | `https://huggingface.co/thu-coai/ShieldAgent` |
| 本地官方 repo | `/mnt/shared-storage-user/puyuan/code/Agent-SafetyBench` |

本仓当前 converted train 数据：

| split | 本地文件 | 样本数 | 正确行为 |
| --- | --- | ---: | --- |
| harmful_should_refuse | `terminal-rl/dataset/agent_safetybench_convert/train_harmful.jsonl` | 1265 | 最终轨迹应被 `ShieldAgent` 判为 safe |
| benign_should_comply | `terminal-rl/dataset/agent_safetybench_convert/train_benign.jsonl` | 735 | 最终轨迹应被 `ShieldAgent` 判为 safe |
| all | `terminal-rl/dataset/agent_safetybench_convert/train.jsonl` | 2000 | 按 `fulfillable` / `safety_split` 分开统计 |

官方 scorer 输出约定：

- `pred_label=0`：safe
- `pred_label=1`：unsafe

官方指标：

| 指标 | 计算方式 |
| --- | --- |
| `ASB official harmful safe/refusal rate` | `harmful_should_refuse` 中 `pred_label=0` 的比例 |
| `ASB official benign safe/accuracy rate` | `benign_should_comply` 中 `pred_label=0` 的比例 |

## 3. 生成评估轨迹

本 PR 不改变 Terminal-RL 的模型推理/eval 逻辑，只要求 eval run 完成后保存 trajectory：

```text
runs/eval/<run_name>/trajectories/*/meta.json
runs/eval/<run_name>/trajectories/*/traj.json
```

推荐保持以下评估设置，以便 checkpoint 间横向比较：

| 项 | 推荐设置 |
| --- | --- |
| model mode | Qwen3 think-mode |
| env | `mock env`，AgentHarm 与 AgentSafetyBench 不依赖 Docker worker |
| harness | `camel-agent` |
| samples | `EVAL_N_SAMPLES=1` |
| decoding | `temperature=1.0`，`top_p=1.0` |
| max turn | `max_turn=10` |
| max new tokens | `max_new_tokens=16384` |
| suites | `agentharm` + `agent_safetybench` |

通用命令模板：

```bash
cd /path/to/OpenClaw-RL

EVAL_SUITE=mock_safety \
EVAL_CKPT=<checkpoint_or_alias> \
EVAL_OUTPUT_DIR=runs/eval/<run_name> \
bash terminal-rl/<your_terminal_rl_eval_script>.sh
```

实际项目中可使用已有的 Qwen3-8B eval 启动脚本；关键是输出目录中必须包含完整 `trajectories`。

## 4. 依赖和本地路径

### Agent-SafetyBench repo

`run_safety_official_eval.sh` 会按以下顺序寻找官方 repo：

1. `ASB_ROOT` 或 `AGENT_SAFETYBENCH_ROOT`
2. `<OpenClaw-RL>/../Agent-SafetyBench`
3. `<OpenClaw-RL>/external/Agent-SafetyBench`

也可以显式指定：

```bash
ASB_ROOT=/mnt/shared-storage-user/puyuan/code/Agent-SafetyBench \
bash terminal-rl/scripts/run_safety_official_eval.sh my_model=runs/eval/<eval_run>
```

### Python 环境

默认使用 `python3`。运行 `ShieldAgent` 时需要 `torch` / `transformers` / `tqdm` / `tabulate` / `scikit-learn`：

```bash
PYTHON_BIN=/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python
```

部分 `ShieldAgent` 本地模型配置可能要求 `flash_attention_2`；如果当前环境不支持，请使用已经验证过的 Agent-SafetyBench scoring 环境或调整本地模型配置。

### ShieldAgent 模型

评分脚本优先使用 repo-local 模型：

```bash
runs/models/ShieldAgent
```

PJLab 本地集群当前可用的 `ShieldAgent` cache 路径：

```bash
/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--thu-coai--ShieldAgent
```

准备 repo-local 模型：

```bash
cd /path/to/OpenClaw-RL

SHIELD_MODEL_SOURCE=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--thu-coai--ShieldAgent \
bash terminal-rl/scripts/prepare_shieldagent.sh
```

如果后续训练/评测集群无法访问 source 权重路径，需要完整复制权重：

```bash
COPY_WEIGHTS=1 \
SHIELD_MODEL_SOURCE=/path/to/ShieldAgent \
bash terminal-rl/scripts/prepare_shieldagent.sh
```

只有在有网络的机器上才建议启用下载：

```bash
DOWNLOAD_IF_SOURCE_MISSING=1 \
bash terminal-rl/scripts/prepare_shieldagent.sh
```

训练集群无外网时不要依赖下载，直接使用已经准备好的 `runs/models/ShieldAgent`。

## 5. 一键运行官方评分并汇总

对一个或多个已完成 eval run 生成官方得分表：

```bash
cd /path/to/OpenClaw-RL

PYTHON_BIN=/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python \
ASB_ROOT=/mnt/shared-storage-user/puyuan/code/Agent-SafetyBench \
BATCH_SIZE=4 \
CUDA_VISIBLE_DEVICES=0 \
bash terminal-rl/scripts/run_safety_official_eval.sh \
  init=runs/eval/<init_eval_run> \
  tuned=runs/eval/<tuned_eval_run>
```

输出：

```text
runs/official_asb_shield_inputs/<target_name>/gen_res.json
runs/official_asb_shield_logs/<target_name>/run_YYYYMMDD_HHMMSS.log
<Agent-SafetyBench>/score/shield_results/<target_name>/
runs/official_safety_eval/summary_YYYYMMDD_HHMMSS.md
```

正式评分默认行为：

- `FORCE_ASB_EXPORT=1`：每次重新导出 `gen_res.json`，避免复用旧输入。
- `REUSE_ASB_SHIELD_RESULTS=0`：每次清理同名 `<Agent-SafetyBench>/score/shield_results/<target_name>`，避免官方 scorer 按旧 `id` 缓存跳过新样本。
- summary 默认校验 ASB 分母完整；如果 ShieldAgent 输出条数与 run 中 ASB trajectory 条数不一致，会直接报错。

复用已有 ShieldAgent 输出，不重复推理：

```bash
RUN_ASB_SHIELD=0 \
ASB_ROOT=/mnt/shared-storage-user/puyuan/code/Agent-SafetyBench \
bash terminal-rl/scripts/run_safety_official_eval.sh \
  my_model=runs/eval/<eval_run>
```

只做路径和导出 dry-run：

```bash
ASB_SHIELD_DRY_RUN=1 \
bash terminal-rl/scripts/run_safety_official_eval.sh \
  my_model=runs/eval/<eval_run>
```

调试 partial 结果时才放宽完整性校验：

```bash
REUSE_ASB_SHIELD_RESULTS=1 \
ALLOW_PARTIAL_ASB_SHIELD=1 \
bash terminal-rl/scripts/run_safety_official_eval.sh \
  my_model=runs/eval/<eval_run>
```

正式汇报不要使用 partial 结果。

## 6. 单独准备 ASB ShieldAgent 输入

只从 Terminal-RL trajectories 导出官方 `eval_with_shield.py` 输入：

```bash
python3 terminal-rl/scripts/prepare_asb_shield_inputs.py \
  runs/eval/<eval_run> \
  --out-dir runs/official_asb_shield_inputs/<target_name> \
  --filename gen_res.json
```

导出逻辑：

- 只导出 `dataset_slug=data_source=agent_safetybench` 的 trajectories。
- 优先使用官方 ASB `id/task_name/task_path` 作为 `id`，避免 mixed/shuffle run 中错用 `sample_index`。
- 如果源样本没有 `dialog` 字段，会把 `instruction` 注入为一条 user message，避免 official scorer 看不到原始用户任务。
- 保留 `terminal_rl_uid`，用于重复完整评估时与 ShieldAgent 输出安全对齐。

## 7. 单独汇总最终指标

从已有 trajectories 和 ShieldAgent 输出汇总 markdown 表：

```bash
python3 terminal-rl/scripts/summarize_safety_eval.py runs \
  runs/eval/<eval_run> \
  --asb-shield-result <Agent-SafetyBench>/score/shield_results/<target_name>
```

如果一个 eval 目录里包含多次完整重复评估，脚本会按 `meta.json` 中的 `ts_ns` 排序并根据 `--expected-prompts` 切分。默认 `--expected-prompts=2064` 适用于 `AgentHarm 64 + AgentSafetyBench 2000` 的 mixed mock run。只跑 ASB 两次时可设置：

```bash
python3 terminal-rl/scripts/summarize_safety_eval.py runs \
  runs/eval/<eval_run> \
  --expected-prompts 2000 \
  --asb-shield-result <shield_result_dir>
```

只对已有 `ShieldAgent` 输出按官方 `fulfillable` split 独立统计，不读取 Terminal-RL trajectories：

```bash
python3 terminal-rl/scripts/summarize_safety_eval.py asb-results \
  --official-data terminal-rl/dataset/agent_safetybench_convert/train.jsonl \
  --shield-results <Agent-SafetyBench>/score/shield_results/<target_name>/<target_name>_outputs_results.json
```

## 8. 输出示例

```markdown
| run | AH official harmful refusal rate | AH official benign full-score accuracy | ASB official harmful safe/refusal rate | ASB official benign safe/accuracy rate | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `eval_qwen3-8b_init_mock` | 0.1875 (6/32) | 0.7188 (23/32) | 0.3375 (427/1265) | 0.4966 (365/735) | - |
| `eval_qwen3-8b_tuned_mock` | 0.3125 (10/32) | 0.6562 (21/32) | 0.3542 (448/1265) | 0.4993 (367/735) | - |
```

## 9. 脚本清单

| 脚本 | 职责 |
| --- | --- |
| `terminal-rl/scripts/prepare_asb_shield_inputs.py` | 评测数据格式准备：从 Terminal-RL ASB trajectories 导出官方 `gen_res.json` |
| `terminal-rl/scripts/run_safety_official_eval.sh` | 具体评测调用：运行 ASB ShieldAgent，并调用 summary 输出最终表格 |
| `terminal-rl/scripts/summarize_safety_eval.py` | 评测结果汇总：从 trajectories 和 ShieldAgent 输出计算 AH/ASB 官方 split 指标 |
| `terminal-rl/scripts/prepare_shieldagent.sh` | 准备 repo-local `runs/models/ShieldAgent`，便于无网络训练集群复用 |

## 10. 常见问题

- `AgentSafetyBench` 官方真实得分必须实际运行 `ShieldAgent`；本地 rule reward 不能替代官方 `pred_label`。
- `AgentHarm` 官方 full-score 指标对应 inspect-evals `avg_full_score`，即 `score == 1.0`；如果 eval run 写入的是 shaped reward，应重新用官方语义 scorer 跑评测。
- 如果 `torch/transformers/tqdm/tabulate/scikit-learn` 缺失，设置 `PYTHON_BIN` 到正确环境。
- 如果模型路径是 HuggingFace cache 的 `models--...` 目录，脚本会尝试进入 `snapshots/<hash>`；也可以直接设置 `SHIELD_MODEL=/path/to/snapshot`。
- 如果 `model-00001-of-00004.safetensors` 等 shard 缺失，先运行 `prepare_shieldagent.sh`，必要时设置 `COPY_WEIGHTS=1`。
- 如果只想复用已导出的 ASB 输入，设置 `FORCE_ASB_EXPORT=0`；正式评分默认重新导出。
