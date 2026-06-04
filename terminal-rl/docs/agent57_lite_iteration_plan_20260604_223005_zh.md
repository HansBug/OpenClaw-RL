# Terminal-RL 中的 Agent57-lite：当前状态 Review 与迭代计划

生成时间：2026-06-04 22:30:05 Asia/Hong_Kong

## 执行摘要

当前 `agent57_lite` 实现是通往 Agent57 风格探索（Agent57-style exploration）的一个有用、低风险桥接版本，但它有意不实现完整 Agent57/NGU。它实现了三个实用组件：

1. 在标量探索权重上进行按 rollout 的 arm 分配。
2. 基于 `action_signature + coarse_observation_fingerprint + exit_code` 计数的轻量级终身新颖性（lifelong novelty）信号。
3. 可选的基于这些 arms 的滑动窗口 UCB（sliding-window UCB），使用最近的类外在 rollout 结果。

这比直接移植 Atari Agent57 更适合 Terminal-RL 的约束：terminal episode 很短（默认 `MAX_TURN=10`），rollout group 较小（默认 `ROLLOUT_BATCH_SIZE=8`、`N_SAMPLES=4`），主要稳定性瓶颈通常是远端 Docker/env 吞吐，而不是纯策略探索。

当前系统与 Agent57/NGU 数学最重要的不匹配点是：当前在 `generate.py` 中以加法组合探索组件，而 NGU 使用乘法形式：

```text
intrinsic = episodic_novelty * lifelong_modulator
```

当前行为：

```text
explore_total = episodic_intrinsic + lifelong_count_bonus + LPRND + CDE + safety_penalty
```

建议下一步：保留当前实现作为稳定 baseline，然后增加可选 `ngu_lite` 组合模式，使用现有 episode signature novelty 作为 episodic 项，使用当前 lifelong counts 作为 modulator。这样可以在不增加 RND 网络、LoRA heads 或额外模型推理的前提下，更接近 Agent57 类比。

## 代码 Baseline

### 当前 Agent57-lite 核心

文件：`terminal-rl/explore_agent57_lite.py`

相关实现：

- `Agent57LiteConfig` 解析环境变量驱动的配置，并暴露 `active`（`enabled or lifelong_enabled or controller != fixed`）。
- `_default_state_path()` 在 `RUN_DIR` 可用时把共享 sqlite state 写到该目录下。
- `_sqlite_next_counts()` 维护全局 lifelong counts 和 `lifelong_traj_seen`。
- `lifelong_keys()` 哈希：

```text
action signature
coarse observation fingerprint
exit-code bucket
```

- `compute_lifelong_bonus()` 返回：

```text
raw = mean(1 / sqrt(count_before + 1))
bonus = beta_for_arm(arm_id) * lifelong_coef * raw
```

并会在 failed/aborted/truncated samples、parse errors、warmup 和 state errors 时抑制奖励。

- `assign_group_arms()` 返回固定 round-robin arms 或 UCB-ranked arms。
- `record_arm_event()` 追加 arm outcome rows，供未来 UCB 选择使用。

### Rollout 集成

文件：`slime/slime/rollout/sglang_rollout.py`

当前行为：

- `_annotate_rollout_groups()` 在 generation 之前为 rollout group 中的每个 sample 分配 `agent57_arm_id`。
- `_apply_agent57_sampling_params()` 可以按 arm 覆盖 `temperature`、`top_p` 和 `top_k`。

这提供了一个无需 LoRA heads 的轻量版 policy-space mixing：arms 通过 scalar beta 和可选 sampling entropy 产生差异。

### Reward 集成

文件：`terminal-rl/generate.py`

当前 reward block：

```text
explore_total =
    intrinsic_scaled
  + safety_penalty
  + lprnd_bonus
  + agent57_lifelong_bonus
  + cde_actor_bonus
```

之后 group 中每个 sample 都会收到相同的 `explore_total`，并加到 `reward["score"]` 上。

这很稳定，也容易推理，但它与 NGU 有两点不同：

1. Lifelong novelty 不调制 episodic novelty；它是单独的加法项。
2. Lifelong count state 会在 eligibility suppression 之前更新，因此 bad attempts 也会消耗 novelty。这对 anti-spam 行为是可辩护的，但它不同于“只奖励有用、可控的探索”。

### 可观测性

文件：`terminal-rl/rollout_log.py`

当前日志包含：

- `terminal/explore/agent57/lifelong_bonus/*`
- `terminal/explore/agent57/lifelong_raw/*`
- `terminal/explore/agent57/lifelong_eligible_rate`
- `terminal/explore/agent57/lifelong_state_error_rate`
- `terminal/explore/agent57/suppressed_ratio/<reason>`
- `terminal/explore/agent57/arm_<id>/*`
- `per_dataset/<dataset>/agent57/*`
- `metrics.jsonl` schema version 4，带 agent57 字段。

这些足以测试当前 Agent57-lite 是否生效、sqlite 是否健康、哪个 arm 占主导，以及 lifelong bonus 是否大多被 warmup/status/parse errors 抑制。

## Terminal-RL 特有约束

### 短 Horizon

默认最大 turn 数是 10。因此 Agent57 面向很长 Atari horizon 的 gamma schedule 在这里作用有限。对 Terminal-RL 来说，在 LoRA/value-head 工作之前实现 arm-specific gamma 不值得。

### Group Sampling 小于原始 Agent57

脚本默认 `ROLLOUT_BATCH_SIZE=8` 和 `N_SAMPLES=4`，debug 时 group 更小。完整 Agent57 使用大量 actors 和长 replay。这里应优先采用简单的 per-group diversity，避免引入需要每个 arm 数百个最近样本的重控制器逻辑。

### Env 吞吐是一等稳定性约束

Rollout generation 会命中远端 CPU workers 和 Docker containers。任何会增加以下内容的探索机制：

- turn count，
- retry rate，
- parse errors，
- terminal command entropy，
- failed reset/close frequency，

即使 RL 信号理论上更好，也可能降低训练质量。因此第一批 Agent57-lite 迭代必须是 reward-only 或 sampling-only，并配套严格 metrics 和小系数。

### Reward Scale 依赖 Dataset

SetA 更像 pass-rate / task success。ASB 和 AgentHarm 使用直接 safety scores。对 SetA 和正负 safety rewards 来说，单一 `success_threshold=0.0` 作为 UCB 标准是合理的，但 mixed runs 最终应支持 dataset-specific UCB success thresholds，或使用 normalized task reward。

## 与 Agent57 / NGU 的差距分析

### 1. Episodic Novelty

当前等价物：

- `generate.py::_explore_intrinsic_bonus()`
- scope 可以是 `episode`
- granularity 可以是 `signature`
- reward 是对 command/tool signatures 的 `sum(1 / sqrt(count))`。

这是对 episodic count novelty 的合理 Terminal-RL 近似。它避免了 embeddings 和 k-NN，这对稳定性是好事。

相对 NGU 缺失：

- 没有 state embedding distance，
- 没有 kernel density，
- 没有 running distance normalization，
- 没有与 controllable representation 的直接连接。

建议立场：暂时不要增加 k-NN embeddings。Terminal observations 是文本化、稀疏且嘈杂的；SimHash/count signatures 更便宜，也更容易 debug。只有在 count-based 方法显示出明确上限后，再考虑 embeddings。

### 2. Lifelong Novelty

当前等价物：

- `explore_agent57_lite.py::lifelong_keys()`
- `raw = mean(1 / sqrt(count_before + 1))`
- sqlite backend 可以跨 rollout workers 共享 state。

这比 RND 更接近 count-based MERCI-style lifelong novelty。它比在 rollout path 中训练 RND predictor 便宜得多，也安全得多。

相对 NGU/RND 缺失：

- 没有 random target network，
- 没有 predictor error，
- 没有 running z-score normalization，
- 没有 modulator 下界 1，
- 没有和 episodic novelty 的乘法耦合。

建议立场：保留 count-based lifelong 作为默认。RND 只应作为单独 research branch，因为它会引入 model-state synchronization、predictor training cost 和额外 failure modes。

### 3. Intrinsic Combination

当前行为是加法。Agent57/NGU 使用：

```text
r_int = r_episode * min(max(alpha_lifelong, 1), L)
```

加法 bonus 更容易调参，但可能奖励“全局新颖但局部无结构”的动作。乘法形式更符合 Agent57 的哲学：episodic novelty 给方向，lifelong novelty 决定是否放大这个方向。

建议下一步实现：

```text
EXPLORE_AGENT57_COMBINE_MODE=add|ngu_lite
EXPLORE_AGENT57_NGU_MOD_CLIP=5.0
EXPLORE_AGENT57_NGU_EPISODIC_SOURCE=signature_intrinsic

episodic = episode signature novelty, before coefficient
life_mod = 1 + min(lifelong_raw, mod_clip - 1)
ngu_bonus = beta * lifelong_coef * episodic * life_mod
```

这保留了 boundedness，并复用已有信号。

### 4. Meta-controller

当前 UCB：

```text
value = mean_success + 0.25 * mean_base - 0.5 * parse_rate - 0.5 * trunc_rate
ucb = value + ucb_c * sqrt(log(total + 1) / n)
```

这是实用的，但并不完全等同于 Agent57。Agent57 使用基于 undiscounted extrinsic episodic return 的 sliding-window UCB，并且包含强制探索。

当前限制：

- 没有 epsilon-forced random arm selection，
- 没有 dataset-aware reward normalization，
- 没有 per-dataset arm statistics，
- UCB 读取 sqlite/local backend，与 lifelong backend 绑定，
- 所有未尝试 arms 得到 infinity，并按 id 排序；这对初始覆盖没问题，但具有确定性。

建议下一步实现：

```text
EXPLORE_AGENT57_UCB_EPSILON=0.2   # start lower than Atari's 0.5
EXPLORE_AGENT57_UCB_MIN_PER_ARM=8
EXPLORE_AGENT57_UCB_VALUE=success|base|normalized_base
EXPLORE_AGENT57_UCB_DATASET_AWARE=1
```

对 Terminal-RL，从 `epsilon=0.1-0.2` 开始，而不是 0.5，因为随机高熵 tool use 可能压垮 CPU workers。

### 5. Universal Value Function / LoRA Heads

当前没有实现，这是有意的。没有 LoRA heads 或基于 `(beta, gamma)` 条件化的 value function 时，arms 并不是真正独立的 policies。它们当前只是：

- 不同 reward weights，
- 可选的不同 sampling parameters，
- 由 group metadata 选择。

这对 reward shaping 实验仍有价值，但不足以声称实现了完整 Agent57 policy-space exploration。

建议立场：在证明无 LoRA 的 Agent57-lite reward signals 稳定之前，推迟 LoRA heads。下一个无 LoRA 步骤应是 arm-specific sampling schedules，而不是 model adapters。

## 推荐迭代计划

### Phase 0：当前 Run 验证

目标：判断当前 Agent57-lite 是否激活且无害。

使用保守配置运行：

```bash
DATASET=seta \
ALGO=dapo \
HARNESS_OPTION=camel-agent \
EXPLORATION_PROFILE=spear_lite \
EXPLORE_INTRINSIC_COEF=0.015 \
EXPLORE_INTRINSIC_DECAY_STEPS=120 \
EXPLORE_ADVANTAGE_BONUS=1 \
EXPLORE_ADVANTAGE_BONUS_COMPONENTS=explore_intrinsic_scaled \
EXPLORE_ADVANTAGE_BONUS_CLIP=0.05 \
EXPLORE_CDE_ACTOR=1 \
EXPLORE_CDE_ACTOR_REWARD_GATE=positive \
EXPLORE_CDE_ACTOR_OMEGA=0.02 \
EXPLORE_CDE_ACTOR_ALPHA=0.05 \
EXPLORE_CDE_ACTOR_KAPPA=4.0 \
EXPLORE_AGENT57_LITE=1 \
EXPLORE_AGENT57_LIFELONG=1 \
EXPLORE_AGENT57_LIFELONG_BACKEND=sqlite \
EXPLORE_AGENT57_CONTROLLER=fixed \
EXPLORE_AGENT57_ARM_BETAS="0,0.002,0.004,0.006,0.008,0.01,0.015,0.02" \
EXPLORE_AGENT57_LIFELONG_COEF=0.005 \
EXPLORE_AGENT57_LIFELONG_WARMUP=64 \
MAX_TURN=10 \
MAX_CKPT_KEEP=2 \
TRAJECTORY_SAVE_INTERVAL=10 \
EXTRA_DAPO_ARGS="--dynamic-sampling-max-groups 64 --dynamic-sampling-max-seconds 1800 --rollout-abort-wait-timeout 300" \
bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

观察这些 metrics：

- `terminal/explore/agent57/lifelong_state_error_rate`：必须保持 0。
- `terminal/explore/agent57/lifelong_eligible_rate`：warmup 后应变为非零。
- `terminal/explore/agent57/suppressed_ratio/warmup`：应在第一个 warmup window 后衰减。
- `terminal/explore/agent57/suppressed_ratio/parse_error`：高值表示探索正在损害 action format。
- `terminal/explore/agent57/arm_<id>/sample_ratio`：fixed controller 应大致均匀覆盖 arms。
- `reward/exploration_ratio`：应保持很小；如果在 SetA 上超过约 `0.1-0.2`，降低系数。

通过条件：

- 没有 worker pressure regression，
- 没有 parse-error spike，
- 没有 state errors，
- exploration reward 相对 task reward 保持较小，
- train/eval task reward 不会在 exploration reward 上升时变平。

### Phase 1：增加 NGU-lite Product Mode

目标：让 intrinsic reward 在不增加新模型的前提下，在数学上更接近 Agent57。

实现思路：

- 增加 `EXPLORE_AGENT57_COMBINE_MODE=add|ngu_lite`。
- 复用 episode signature novelty 作为 `r_ep`。
- 复用 lifelong count novelty 作为 `raw_life`。
- 计算：

```text
life_mod = min(max(1 + raw_life, 1), EXPLORE_AGENT57_NGU_MOD_CLIP)
ngu_bonus = beta * lifelong_coef * r_ep * life_mod
```

集成细节：

- 在 `add` 模式中，保持当前行为。
- 在 `ngu_lite` 模式中，避免 double counting，方式可以是：
  - 不再把 `_intr_scaled` 单独加到 `explore_total`，或
  - 只把 `ngu_bonus` 作为 post-normalization advantage component。

推荐测试默认值：

```bash
EXPLORE_AGENT57_COMBINE_MODE=ngu_lite
EXPLORE_AGENT57_NGU_MOD_CLIP=3
EXPLORE_AGENT57_LIFELONG_COEF=0.002
EXPLORE_INTRINSIC_COEF=0.01
```

风险：

- Product mode 可能放大重复 multi-action rollouts；用 per-rollout `EXPLORE_AGENT57_MAX_BONUS=0.03-0.05` 做 cap。

### Phase 2：改进面向 Terminal-RL 的 UCB

目标：让 meta-controller 在不扰乱 rollout sampling 的前提下变得有用。

推荐改动：

1. 增加 epsilon-forced exploration：

```text
EXPLORE_AGENT57_UCB_EPSILON=0.1
```

2. 在 ranking 前增加每个 arm 的最小样本数：

```text
EXPLORE_AGENT57_UCB_MIN_PER_ARM=8
```

3. 在 `arm_events` 中记录 dataset name，然后为 mixed SetA/ASB/AH runs 计算 per-dataset UCB stats。

4. 对 UCB 使用 normalized base reward：

```text
seta: pass/fail or accuracy in [0, 1]
agent_safetybench/agentharm: clipped direct score mapped to comparable range
```

5. 对所有 UCB runs 保留 `EXPLORE_AGENT57_KEEP_BASELINE=1`，确保每个 group 保留一个低探索 sample。

推荐测试配置：

```bash
EXPLORE_AGENT57_CONTROLLER=ucb
EXPLORE_AGENT57_UCB_EPSILON=0.1
EXPLORE_AGENT57_UCB_WINDOW=256
EXPLORE_AGENT57_UCB_C=0.25
EXPLORE_AGENT57_KEEP_BASELINE=1
```

不要在 Phase 0 确认 fixed arms 稳定之前启用 UCB。

### Phase 3：在 Embeddings 之前改进 State Signatures

目标：在不增加神经 RND 的情况下改善 novelty 质量。

当前 lifelong key 是 action + coarse output + exit code。它很好，但可能漏掉 task-context differences。增加可选字段：

- dataset name，
- task id/path bucket，
- current turn index bucket，
- command family，
- whether a test command was run，
- whether file modifications occurred。

可能的 env：

```text
EXPLORE_AGENT57_LIFELONG_KEY_VERSION=v1|v2
EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK=0|1
EXPLORE_AGENT57_LIFELONG_INCLUDE_TURN=0|1
```

权衡：

- 包含 task path 会阻止 novelty 跨任务泛化。
- 排除 task path 可能把无关任务压到同一个 action bucket。

推荐默认：

- SetA only：排除精确 task id，包含 dataset 和 command family。
- Mixed safety runs：包含 dataset 和 split，不包含精确 task id。

### Phase 4：可选 Embedding/k-NN Research Branch

只有在 count-based NGU-lite 稳定后再做：

1. 从以下信息构建 text-state signature：
   - tool command，
   - compressed stdout/stderr，
   - exit code，
   - task metadata。

2. 使用 frozen small model 或 deterministic SimHash 编码。

3. 使用 per-episode memory 加 approximate k-NN 或 Hamming distance。

4. 保持离线或通过以下开关控制：

```text
EXPLORE_AGENT57_EPISODIC_BACKEND=count|simhash_knn
```

先避免 online RND。RND 会引入 predictor training、同步和额外非确定性，而当前代码库的主要操作风险是 Docker env 稳定性。

### Phase 5：LoRA Heads / 真正的 Policy-Space Mixing

这是最接近 Agent57 universal value function 的类比，但它是更大的系统改动。

前置条件：

- 当前 Agent57-lite reward metrics 稳定，
- UCB 不会让 arm coverage collapse，
- CPU-worker close/reset path 稳定，
- W&B 显示没有 parse-error 或 truncation regression。

然后实现：

- 先做 K=4，不是 K=8。
- 每个 head 具有：
  - beta，
  - sampling temperature/top-p，
  - optional prompt style。
- 只在 uniform arm sampling warmup stage 后使用 UCB。

在有 turn-level value 或 credit assignment 机制之前，不要实现 gamma-specific heads；Terminal-RL 的 horizon 太短，gamma 不应是第一个 lever。

## 具体下一批代码改动

推荐下一 PR，按顺序：

1. 增加默认 `add` 的 `EXPLORE_AGENT57_COMBINE_MODE`。
2. 增加 `ngu_lite` product mode，使用当前 episode signature novelty 和 lifelong count raw。
3. 增加 per-rollout `EXPLORE_AGENT57_MAX_BONUS` clamp。
4. 增加 UCB epsilon 和 minimum-arm warmup，但默认保持 inactive。
5. 扩展 `arm_events`，加入 dataset 和可能的 task source，为 future dataset-aware UCB 做准备。
6. 增加小工具 `terminal-rl/scripts/analyze_agent57_lite.py`，读取 `logs/metrics.jsonl` 并打印：
   - arm coverage，
   - top suppressed reasons，
   - state error rate，
   - exploration/task reward ratio，
   - per-arm reward deltas。

## 推荐默认值

稳定加法模式：

```bash
EXPLORE_AGENT57_LITE=1
EXPLORE_AGENT57_LIFELONG=1
EXPLORE_AGENT57_LIFELONG_BACKEND=sqlite
EXPLORE_AGENT57_CONTROLLER=fixed
EXPLORE_AGENT57_ARM_BETAS="0,0.002,0.004,0.006,0.008,0.01,0.015,0.02"
EXPLORE_AGENT57_LIFELONG_COEF=0.005
EXPLORE_AGENT57_LIFELONG_WARMUP=64
```

实现后首次 NGU-lite 测试：

```bash
EXPLORE_AGENT57_COMBINE_MODE=ngu_lite
EXPLORE_AGENT57_NGU_MOD_CLIP=3
EXPLORE_AGENT57_MAX_BONUS=0.05
EXPLORE_AGENT57_CONTROLLER=fixed
EXPLORE_AGENT57_LIFELONG_COEF=0.002
EXPLORE_INTRINSIC_SCOPE=episode
EXPLORE_INTRINSIC_GRANULARITY=signature
```

fixed-mode 稳定后的首次 UCB 测试：

```bash
EXPLORE_AGENT57_CONTROLLER=ucb
EXPLORE_AGENT57_UCB_EPSILON=0.1
EXPLORE_AGENT57_UCB_C=0.25
EXPLORE_AGENT57_UCB_WINDOW=256
EXPLORE_AGENT57_KEEP_BASELINE=1
```

## Stop Conditions

如果出现以下任一情况，禁用或降低 Agent57-lite：

- `lifelong_state_error_rate > 0.01`
- warmup 后 `parse_error` suppression 占主导
- `reward/exploration_ratio > 0.2` 且 task reward 持平或下降
- CPU worker pending closes 或 reset failures 增加
- UCB arm coverage 过早 collapse 到单个 non-baseline arm
- shaped rewards 让 group variance 变差，dynamic sampling aborts 更多 groups

## 结论

当前实现是一个保守且操作兼容的 Agent57-lite scaffold。最好把它看成：

```text
Agent57-inspired count novelty + scalar arms + optional UCB
```

而不是完整 Agent57。最高价值的下一次迭代不是 RND 或 LoRA heads；而是可选的 NGU-lite product mode，把当前 episodic 和 lifelong signals 转换成 Agent57 使用的相同方向结构：

```text
episode novelty chooses where to explore;
lifelong novelty decides how much to amplify it.
```

这个改动很小、可逆，并且可以直接用现有 W&B/JSONL metrics 测试。
