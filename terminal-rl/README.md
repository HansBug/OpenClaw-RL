# Terminal-RL Exploration 技术文档

> 面向接手同事的中文说明，基于当前代码实现（`generate.py`、`terminal-rl_qwen3-8b_exploration_pu.sh`、`terminal-rl_qwen3-8b_pu.sh`）撰写。原 `EXPLORATION.md` 与 `LAMER_AGENT57_INTEGRATION.md` 保留以供参考，但实际行为以本文档和代码为准。

---

## 一、概览

本模块在 Terminal-RL（基于 slime + Megatron-LM 的 GRPO 在线强化学习框架）之上，叠加多种 **探索增强（Exploration Bonus）** 与训练超参覆盖，目的是改善 LLM Agent 在稀疏奖励终端任务（Terminal-Bench、SETA、CTF、Agent-SafetyBench）上的样本效率与覆盖广度。

核心思路借鉴：

| 工作 | 借鉴点 |
|------|--------|
| **MERCI** (Count Counts) | 1/√N count-based 内在奖励的整体结构 |
| **Agent57** (DeepMind 2020) | 子目标粒度新颖性、生命周期价值的概念 |
| **LaMer** (ICLR '26) | 多次尝试 + 反思（接口已就绪，实际重启需 `agent_runner` 支持） |
| **AEPO** | 熵 bonus 防止 mode collapse |

**设计约束**：所有探索功能默认全部 **关闭**；只要不设任何 `EXPLORE_*` 环境变量，wrapper 的行为与原 baseline 字节级等价（直接 `exec` 到主脚本）。

本模块共暴露 **14 个 `EXPLORE_*` 环境变量**，分别控制 8 类增强（见 §五速查表）。

---

## 二、Baseline 介绍

### 2.1 框架栈

- **训练框架**：[slime](https://github.com/THUDM/slime)（GRPO 异步在线 RL）+ Megatron-LM（Tensor Parallel）。
- **模型**：Qwen3-8B（默认 `HF_CKPT=.../slime/Qwen3-8B/`），4–8 GPU。
- **Rollout 架构**：CPU worker 上跑 `pool_server`（DooD docker），GPU worker 上跑 actor + sglang rollout。
- **任务集**：`DATASET` 环境变量切换 `seta` / `safety` / `mixed`。

### 2.2 关键文件

```
terminal-rl/
├── terminal-rl_qwen3-8b_pu.sh              # 主训练脚本（baseline）
├── terminal-rl_qwen3-8b_exploration_pu.sh  # 探索 wrapper（exec 上面那个）
├── generate.py                             # rollout + 奖励合成（已注入探索代码）
├── configs/
│   ├── rollout_qwen3.yaml                  # 默认 rollout 配置
│   └── rollout_qwen3_think.yaml            # Qwen3 think-mode 配置
├── README.md                               # 项目使用指南（数据集 + 启动）
├── README_EXPLORATION.md                   # ← 本文件
├── EXPLORATION.md                          # （历史文档，保留）
└── LAMER_AGENT57_INTEGRATION.md            # （历史文档，保留）
```

### 2.3 Baseline 奖励结构（`generate.py` 中 `_build_samples`）

```
final_score = discounted_base                 # outcome reward, 2*acc-1
            + prm_coef * prm_turn_score       # 可选：PRM 评判
            + safety_coef * safety_val        # 可选：ClawSentry 安全奖励
```

本探索模块在 `_build_samples` 之后再追加：

```
final_score += EXPLORE_INTRINSIC_COEF * intrinsic_bonus
            + safety_penalty                  # 危险命令负惩罚
            + EXPLORE_LPRND_COEF * lprnd_bonus
```

这些字段会写入 `runs/{run_id}/trajectories/.../meta.json` 的 `reward` 子字典，便于事后归因。

---

## 三、Exploration 技术原理

### 3.1 Entropy Bonus（AEPO 风格）

通过 slime 原生 `--entropy-coef` 参数加入 entropy loss：

$$\mathcal{L} = \mathcal{L}_\text{GRPO} - \beta_{ent}\,H(\pi_\theta)$$

防止 mode collapse（baseline 默认 `entropy_coef=0`，曾观察到 entropy 在某些 rollout 前坍缩至 0）。

### 3.2 Think Mode（Qwen3 CoT）

切换 rollout 配置文件至 `configs/rollout_qwen3_think.yaml`，将 `non_think_mode: false`，启用 Qwen3 原生 `<think>...</think>` CoT。

### 3.3 Rollout 温度覆盖

通过环境变量 `ROLLOUT_TEMPERATURE` 覆盖 baseline 的 `--rollout-temperature 1`，提升采样多样性。

### 3.4 Count-based 内在奖励（MERCI 简化版）

$$r^\text{intr} = \sum_{i \in \text{turns}} \frac{1}{\sqrt{N(s_i)}}$$

其中 $N(s_i)$ 为命令 $s_i$ 在进程级计数器中的累积访问次数。

**两种粒度**（`EXPLORE_INTRINSIC_GRANULARITY`）：

- `raw`（默认）：完整命令字符串 MD5。
- `signature`：仅取 `cmd名 | arg1 | arg2`（`shlex.split(cmd)[:3]`），将近义写法（如 `ls -la /tmp` 与 `ls -al /tmp/`）归入同桶 —— 对应 Agent57 子目标粒度。

与 MERCI 原版的关系：MERCI 用 Coin Flipping Network 估计 token-level pseudo-count；本实现是其 **简化版**，用确定性哈希在 command 粒度计数，零额外参数。

### 3.5 LP-RND 生命周期新颖性（草案 C）

复用 slime 已计算的 `output_token_logprobs`，无需额外前向传播：

$$\bar\ell = \frac{1}{T}\sum_t \log\pi_\theta(a_t | s_t),\quad r^\text{lprnd} = \text{clip}\!\left(\frac{-\bar\ell-\mu}{\sigma},\ 0,\ L\right)$$

- $\mu, \sigma$ 由 **Welford 在线算法** 维护，进程级。
- 前 32 条轨迹为 warmup，期间奖励恒为 0。
- $L$ 由 `EXPLORE_LPRND_CLIP` 控制，默认 3.0。

**直觉**：当前策略对某轨迹"惊讶"（mean negative-logprob 高）→ 探索到低密度区域 → 给正奖励。Agent57 RND 需要额外 random network；LP-RND 用 policy 自身的对数概率作信号，零参数。

### 3.6 安全预过滤惩罚

对每个 turn 的命令做正则匹配，命中危险模式则施加负奖励 `EXPLORE_SAFETY_FILTER_COEF`（默认 −0.5）。

匹配模式：`rm -rf /` 系列、`curl|bash` 注入、`chmod 777 /`、写 `/etc/passwd|shadow|sudoers`、读 `/etc/shadow`、fork bomb。

与 ClawSentry 安全奖励正交：前者是 reward shaping，本机制是命令级硬惩罚。

### 3.7 多次尝试反思（LaMer，旗标已就绪，env restart 待实现）

`EXPLORE_RETRY_ATTEMPTS > 1` 时透传环境变量并打印 `[WARN] Multi-attempt requires agent_runner support (not yet implemented)`。

后续 P1：在 `agent_runner.py` 的 run 循环里检测 `EXPLORE_RETRY_ATTEMPTS`，失败时追加 reflection turn 并 `env_client.reset()`；按 `EXPLORE_RETRY_TRAJ_GAMMA^attempt_idx` 折扣奖励。

### 3.8 MAX_TURN 覆盖

通过 `EXPLORE_MAX_TURN` 覆盖 baseline 默认的 `max_iteration=10`。脚本生成 per-run yaml 覆盖配置。

---

## 四、具体实现介绍

### 4.1 `generate.py` 关键代码段

**模块级常量**（约 line 42–80）：

```python
# Count-based
_EXPLORE_INTRINSIC_ENABLED = os.getenv("EXPLORE_INTRINSIC_ENABLED", "0") == "1"
_EXPLORE_INTRINSIC_COEF = float(os.getenv("EXPLORE_INTRINSIC_COEF", "0.1"))
_EXPLORE_INTRINSIC_GRANULARITY = os.getenv("EXPLORE_INTRINSIC_GRANULARITY", "raw")
_CMD_COUNTER: Dict[str, int] = {}

# LP-RND
_EXPLORE_LPRND_ENABLED = os.getenv("EXPLORE_LPRND_ENABLED", "0") == "1"
_EXPLORE_LPRND_COEF = float(os.getenv("EXPLORE_LPRND_COEF", "0.05"))
_EXPLORE_LPRND_CLIP = float(os.getenv("EXPLORE_LPRND_CLIP", "3.0"))
_LPRND_STATS = {"n": 0, "mean": 0.0, "m2": 0.0}

# Retry (signal-only)
_EXPLORE_RETRY_ATTEMPTS = int(os.getenv("EXPLORE_RETRY_ATTEMPTS", "1"))
_EXPLORE_RETRY_TRAJ_GAMMA = float(os.getenv("EXPLORE_RETRY_TRAJ_GAMMA", "1.0"))

# Safety pre-filter
_EXPLORE_SAFETY_FILTER_ENABLED = os.getenv("EXPLORE_SAFETY_FILTER_ENABLED", "0") == "1"
_EXPLORE_SAFETY_FILTER_COEF = float(os.getenv("EXPLORE_SAFETY_FILTER_COEF", "-0.5"))
_DANGER_RE = re.compile(...)
```

**辅助函数**：

| 函数 | 行号 | 作用 |
|------|------|------|
| `_cmd_signature(cmd)` | ~83 | shlex 切分，取前 3 个 token（cmd + 2 args），拼成签名键 |
| `_explore_intrinsic_bonus(turn_records)` | ~101 | 遍历 turn，按 `EXPLORE_INTRINSIC_GRANULARITY` 选择哈希源，更新 `_CMD_COUNTER`，累加 1/√N |
| `_explore_safety_penalty(turn_records)` | ~125 | 对每条 cmd 做正则匹配，命中则累加 `EXPLORE_SAFETY_FILTER_COEF` |
| `_explore_lprnd_bonus(interactions)` | ~137 | 提取 `output_token_logprobs`，算 mean negative logprob，Welford 归一化 + clip |

**调用点**（约 line 1078–1088，在 `_build_samples` 之后）：

```python
if _EXPLORE_INTRINSIC_ENABLED or _EXPLORE_SAFETY_FILTER_ENABLED or _EXPLORE_LPRND_ENABLED:
    _intr_bonus  = _explore_intrinsic_bonus(turn_records)
    _safe_penalty = _explore_safety_penalty(turn_records)
    _lprnd_bonus = _explore_lprnd_bonus(interactions) * _EXPLORE_LPRND_COEF
    for s in samples:
        if isinstance(s.reward, dict) and "score" in s.reward:
            s.reward["score"] += (_intr_bonus * _EXPLORE_INTRINSIC_COEF
                                  + _safe_penalty + _lprnd_bonus)
            s.reward["explore_intrinsic"]      = _intr_bonus
            s.reward["explore_safety_penalty"] = _safe_penalty
            s.reward["explore_lprnd"]          = _lprnd_bonus
```

### 4.2 `terminal-rl_qwen3-8b_exploration_pu.sh` 关键段

纯 wrapper 设计：读取所有 `EXPLORE_*` 环境变量 → 转换为内部 env vars → 拼装 `RUN_ID` 后缀 → `exec bash terminal-rl_qwen3-8b_pu.sh`。

关键路由：

| 用户输入 | wrapper 行为 |
|----------|-------------|
| `EXPLORE_ENTROPY_COEF=0.01` | `EXTRA_GRPO_ARGS="--entropy-coef 0.01"` |
| `EXPLORE_THINK_MODE=1` | `CUSTOM_CONFIG_PATH=configs/rollout_qwen3_think.yaml` |
| `EXPLORE_TEMP_HIGH=1.2` | `ROLLOUT_TEMPERATURE=1.2` |
| `EXPLORE_INTRINSIC=1` | `EXPLORE_INTRINSIC_ENABLED=1` |
| `EXPLORE_LPRND=1` | `EXPLORE_LPRND_ENABLED=1` |
| `EXPLORE_SAFETY_FILTER=1` | `EXPLORE_SAFETY_FILTER_ENABLED=1` |
| `EXPLORE_MAX_TURN=15` | `MAX_TURN=15`（覆盖 baseline 默认 10） |
| `EXPLORE_RETRY_ATTEMPTS=3` | 透传 env + 打印 WARN（待 agent_runner 实现） |

### 4.3 `terminal-rl_qwen3-8b_pu.sh` 微改

仅 2 处改动让 wrapper 能透传环境变量：

- L409：`--rollout-temperature "${ROLLOUT_TEMPERATURE:-1}"` （原硬编码 `1`）
- L700：在 `${GRPO_ARGS[@]}` 之后追加 `${EXTRA_GRPO_ARGS:-} \`

两处默认行为不变。

---

## 五、环境变量速查表

| 变量 | 默认值 | 类型 | 作用 |
|------|--------|------|------|
| `EXPLORE_ENTROPY_COEF` | `0.0` | float | AEPO 熵 bonus 系数；非 0 时透传为 `--entropy-coef X` |
| `EXPLORE_THINK_MODE` | `0` | bool | Qwen3 CoT think mode（切换 rollout yaml） |
| `EXPLORE_TEMP_HIGH` | *（空）* | float | rollout 温度覆盖（空=继承 baseline 1.0） |
| `EXPLORE_INTRINSIC` | `0` | bool | Count-based 内在奖励总开关 |
| `EXPLORE_INTRINSIC_COEF` | `0.1` | float | 内在奖励权重 |
| `EXPLORE_INTRINSIC_GRANULARITY` | `raw` | str | `raw` / `signature` 二选一 |
| `EXPLORE_LPRND` | `0` | bool | LP-RND 生命周期新颖性开关 |
| `EXPLORE_LPRND_COEF` | `0.05` | float | LP-RND 权重 |
| `EXPLORE_LPRND_CLIP` | `3.0` | float | z-score 裁剪上限 |
| `EXPLORE_SAFETY_FILTER` | `0` | bool | 危险命令正则惩罚开关 |
| `EXPLORE_SAFETY_FILTER_COEF` | `-0.5` | float | 危险命令惩罚值（负数） |
| `EXPLORE_RETRY_ATTEMPTS` | `1` | int | 失败轨迹重试次数；当前仅透传，实际重启逻辑待实现 |
| `EXPLORE_RETRY_TRAJ_GAMMA` | `1.0` | float | 跨 attempt 奖励折扣（LaMer 用 0.6） |
| `EXPLORE_MAX_TURN` | *（空）* | int | 覆盖 `max_iteration`（baseline 默认 10） |

---

## 六、实验测试命令

以下命令均假定已 `cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL` 且 `WORKER_URLS` 已 export。

### 6.1 兼容性验证（baseline 对照）

所有 `EXPLORE_*` 不设，应与直接跑 baseline 完全等价：

```bash
WORKER_URLS=http://cpu-worker:18081 DEBUG_MODE=1 \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# 对照
WORKER_URLS=http://cpu-worker:18081 DEBUG_MODE=1 \
  bash terminal-rl/terminal-rl_qwen3-8b_pu.sh
# 期望：runs/<id>/config/run_config.json 内容除时间戳外完全一致
```

### 6.2 单技术消融

```bash
# (a) 仅 entropy bonus
EXPLORE_ENTROPY_COEF=0.01 WORKER_URLS=... \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# (b) 仅 think mode
EXPLORE_THINK_MODE=1 WORKER_URLS=... \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# (c) 仅 count-based intrinsic (raw 粒度)
EXPLORE_INTRINSIC=1 EXPLORE_INTRINSIC_COEF=0.1 \
  WORKER_URLS=... bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# (d) 仅 count-based intrinsic (signature 粒度, Agent57 风格)
EXPLORE_INTRINSIC=1 EXPLORE_INTRINSIC_COEF=0.1 \
  EXPLORE_INTRINSIC_GRANULARITY=signature \
  WORKER_URLS=... bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# (e) 仅 LP-RND
EXPLORE_LPRND=1 EXPLORE_LPRND_COEF=0.05 \
  WORKER_URLS=... bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# (f) 仅 safety filter
EXPLORE_SAFETY_FILTER=1 EXPLORE_SAFETY_FILTER_COEF=-0.5 \
  WORKER_URLS=... bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

### 6.3 推荐配置（v2 立即可用全栈）

```bash
EXPLORE_ENTROPY_COEF=0.01 \
EXPLORE_THINK_MODE=1 \
EXPLORE_INTRINSIC=1 \
EXPLORE_INTRINSIC_GRANULARITY=signature \
EXPLORE_LPRND=1 \
EXPLORE_LPRND_COEF=0.05 \
EXPLORE_MAX_TURN=15 \
WORKER_URLS=http://cpu-worker:18081 \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
# 期望 RUN_ID 后缀：_explore_ent0.01_think_int_lprnd_<ts>
```

### 6.4 离线单元测试（无需 GPU/CPU worker）

```bash
# 语法检查
python3 -m py_compile terminal-rl/generate.py && echo "py OK"
bash -n  terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh && echo "sh OK"

# 内在奖励逻辑最小烟雾测试（不依赖 generate.py 全部依赖）
python3 - <<'PY'
import hashlib, math, shlex
ctr = {}
def bonus(cmds):
    t = 0.0
    for c in cmds:
        sig = "|".join(shlex.split(c)[:3]) if c.strip() else "__empty__"
        k = hashlib.md5(sig.encode()).hexdigest()[:10]
        ctr[k] = ctr.get(k, 0) + 1
        t += 1.0 / math.sqrt(ctr[k])
    return t

print("first  ls -la /tmp:",  bonus(["ls -la /tmp"]))   # ~1.0
print("second ls -al /tmp/:", bonus(["ls -al /tmp/"]))  # ~0.707 (signature 命中同桶)
print("third  ls -la /etc:",  bonus(["ls -la /etc"]))   # ~1.0 (不同桶)
PY
```

### 6.5 验收检查清单

训练启动后请确认：

- [ ] `runs/<run_id>/config/run_config.json` 字段符合预期。
- [ ] `runs/<run_id>/logs/train.log` 出现 `[explore] xxx ON` 日志行。
- [ ] 一旦有 rollout 完成，`runs/<run_id>/trajectories/<task>__g0__i0__*/meta.json` 的 `reward` 字段含 `explore_intrinsic` / `explore_lprnd` / `explore_safety_penalty`。
- [ ] 若 `EXPLORE_LPRND=1`，**前 32 条** rollout 的 `explore_lprnd` 应恒为 0（warmup），第 33 条起才有正值。
- [ ] `wandb`（如启用）能看到 `reward/explore_intrinsic_mean` 等自定义 metric。

---

## 七、已知 Bug 与修复建议（对照 MERCI / SPEAR 源码审查）

参考 `/mnt/shared-storage-user/puyuan/code/MERCI/` 与 `/mnt/shared-storage-user/puyuan/code/SPEAR/` 的实现，本模块当前存在以下问题。优先级 P0 建议尽快修；P1 / P2 可纳入下一迭代。

### Bug 1 [P0]：`_CMD_COUNTER` / `_LPRND_STATS` 进程隔离

**问题**：两者均为模块级 Python dict，slime 异步 rollout（多个 sglang worker / Ray actor）下，每个 worker 进程独立维护一份计数器和 Welford 统计。结果：
- 同一命令在 8 个 worker 上独立累积 N，1/√N 奖励不一致。
- LP-RND 的 μ/σ 在各 worker 上漂移方向不同，归一化基准不同步。

**对照 MERCI**：把 pseudo-count 估计放进全局神经网络 **Coin Flipping Network**，并通过 RL trainer 的 `exploration_model` 子模块跨 worker 同步参数（见 `MERCI/recipe/dapo/example/run_qwen2.5_math_dapo_cfn.sh` 的 `exploration_model.model.pretrain_path`），从根本上避开进程隔离问题。

**修复建议**：

- **短期 workaround**：放弃跨轨迹比较，把 `_CMD_COUNTER` / `_LPRND_STATS` 改为 **per-rollout reset**（每次 `generate` 入口清空）。优点：消除不一致，无需 IPC。缺点：丢失"跨轨迹"探索信号，退化为 episode-internal 新颖性。
- **中期**：用 Ray Actor 维护一个全局 `CounterServer`，rollout 通过 `ray.get` 同步更新；或写入 Redis。
- **长期**：参考 MERCI，把计数器换成轻量级 CFN（约 1M 参数），跨 worker 通过 broadcast 同步权重。

### Bug 2 [P0]：LP-RND warmup 期间仍更新统计

**位置**：`_explore_lprnd_bonus`（generate.py 约 L137）。

```python
# Welford 更新（先做）
s["n"] += 1
delta = surprise - s["mean"]
s["mean"] += delta / s["n"]
s["m2"] += delta * (surprise - s["mean"])
if s["n"] < 32:
    return 0.0  # warmup，但统计已被更新
```

**问题**：warmup 期前 32 条轨迹（训练初期 entropy 最高、最具探索价值）的 surprise 已纳入 μ/σ；之后归一化时它们作为基线，会让后续轨迹的 z-score 显著偏低，削弱奖励信号。

**修复建议**（推荐方案 A）：

```python
# 方案 A：warmup 期不更新统计，只计数
if s["n"] < 32:
    s["n"] += 1
    return 0.0
# 通过 warmup 后再做完整 Welford
...
```

或方案 B：保持先更新统计，但延长 warmup 至 256，让初期数据被稀释。

### Bug 3 [P1]：`_cmd_signature` 对空命令未保护

**位置**：generate.py 约 L83。

```python
parts = shlex.split(cmd)[:3]
return "|".join(parts)
```

- `shlex.split("")` → `[]` → `"|".join([]) == ""` → 所有空命令共享同一桶。
- `shlex.split` 在含未配对引号的命令上抛 `ValueError`（已有 `except Exception: return cmd[:80]` 保护，OK）。

注：上游 `_explore_intrinsic_bonus` 已有 `if not cmd: continue` 保护，**当前不会触发**；但 `_cmd_signature` 作为独立 helper 一旦被复用就脆弱。

**修复建议**：

```python
def _cmd_signature(cmd: str) -> str:
    if not cmd or not cmd.strip():
        return "__empty__"
    ...
```

### Bug 4 [P1]：safety 正则未覆盖反引号 / `$()` 子 shell 注入

现有正则只匹配字面 `rm -rf /`、`curl|bash` 等。LLM 若学会通过 `eval $(echo "rm -rf /")` 或 `\$(printf 'rm -rf /')` 间接执行，可绕过。当前威胁模型下风险较低（LLM 主动越狱），但属于深度防御缺口。

**修复建议**：低优先级，可在 P2 引入 `bashlex` AST 解析；或为 ClawSentry 提供命令文本，让其 LLM-judge 做语义判断。

### Bug 5 [P2]：LP-RND 缺少 SPEAR 式的 "intrinsic reward decay"

**对照 SPEAR**：SPEAR 在内在奖励上设计了 **curriculum decay**：早期权重大（鼓励探索），后期权重渐弱（鼓励 exploitation 已发现的成功轨迹）。我们目前的 `EXPLORE_LPRND_COEF` / `EXPLORE_INTRINSIC_COEF` 都是静态常数。

**修复建议**：增加 `EXPLORE_*_DECAY_STEPS`（默认 0=关闭），训练 step ≥ 该值后线性衰减到 0。可参考 SPEAR `recipe/spear/` 下的实现。

### Bug 6 [P2]：`_explore_intrinsic_bonus` 缺少 "per-group 归一化"

**问题**：GRPO 用 group-internal mean baseline 计算 advantage；当前 intrinsic bonus 直接叠加在 raw reward 上，**会被 baseline 减掉**。同一 prompt 的 8 个 rollout 通常用相似命令，其 intrinsic bonus 高度相关，减完之后剩下的 explore signal 较弱。

**修复建议**：低优先级，可考虑：
- 让 intrinsic bonus 在 advantage 计算之前先减去 group mean，再乘 coef。
- 或者直接把 intrinsic bonus 加在 advantage 上，绕过 GRPO baseline。

MERCI 论文未深入讨论该细节，需做消融。

### Bug 7 [P2]：多次尝试反思（LaMer）未实际接入

`EXPLORE_RETRY_ATTEMPTS=3` 只透传环境变量，`agent_runner.py` 未读取，行为与 `=1` 等价。当前仅打印 WARN。

**修复建议**：见 §3.7 P1 计划；需在 `agent_runner` 主循环增加 "detect failure → inject reflection → reset env → replay" 三段逻辑。预估 ~200 行。

---

## 八、与上游工作的关系小结

| 模块 | 直接复刻 | 简化适配 | 创新 |
|------|---------|---------|------|
| Count-based bonus | MERCI 的整体公式 (1/√N) | 用确定性 hash 替代 CFN | `signature` 粒度 |
| LP-RND | — | RND 思想 | 复用 logprob 作 surprise，零参数 |
| Safety filter | — | — | 与 ClawSentry 正交的命令级硬惩罚 |
| Multi-attempt | LaMer 概念 | 仅旗标，待实现 | — |
| Entropy bonus | AEPO | slime 原生 `--entropy-coef` 直接打开 | — |

本模块定位是 **轻量化、可消融、零侵入** 的探索工具箱，**不试图复刻 MERCI / SPEAR / LaMer 全部细节**。如需更激进的探索（CFN 跨 worker 同步、SPEAR 的 self-imitation replay buffer），需要更深入的 slime trainer 改造。
