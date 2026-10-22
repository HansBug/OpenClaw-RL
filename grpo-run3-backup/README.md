# GRPO run-3 local patches (backup, 2026-10-22)

The terminal-RL Binary GRPO run-3 (qwen3-8b, wandb `msp60ius`, issue #4) was
launched from upstream `Gen-Verse/OpenClaw-RL@a58ec520` plus local changes that
were never committed. This directory preserves them.

- `patches/mbridge-0.15.1-qwen2.patch`: the change to the installed
  `mbridge==0.15.1` file `mbridge/models/qwen2.py`, found on the 34.48 host in
  the `tbench-rl` conda env (its content differs from the pip RECORD hash).
  Apply it to a clean `mbridge==0.15.1` install with `patch -p0`. It adds the
  local (non-TE) implementation mapping for the pre-MLP and post-attention
  layernorms (issue #2, "补 local-impl mapping").
- `launch/run_qwen3_8b_experiment.sh`: the launcher from gist
  `5f01a4a703b55d7991a00bd07e66509a`. It is a copy of
  `terminal-rl/terminal-rl_qwen3-8b.sh` with three compat additions
  (conda env activation, sourcing a `.env` for WANDB vars, and wrapper changes).
  It reads credentials from the environment and contains no secret values.

Caveats: the four wrapper compat changes described in issue #2 are only
recoverable from this launcher script; the base commit is upstream, not a fork.
