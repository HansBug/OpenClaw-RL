# TBv2.1 evaluation scripts (archived 2026-10-21)

Snapshot of the scripts used to run the Terminal-Bench 2.1 (Harbor, terminus-2) evaluations of Qwen3-8B checkpoints in July 2026
(base / outcome-gate iter299 / DAPO baseline iter899 / iter1099; see HansBug/OpenClaw-RL issues #25, #27, #31, #32).

- Origin: `narmodel/zhangshaoang/tbv2.1/{bin,patches,docs,configs,README.md,env.sh}` on the Shanghai AI Lab cluster (not a git repository there).
- Included: text files under 2 MB only. Excluded: `__pycache__`, `*.pyc`, `*.tar.gz` and large binaries/runtimes (rootlesskit, python_sglang, docker data, job outputs).
- Checked for credentials before publishing (token/key/URL-password patterns): none found.
- Paths inside the scripts refer to the lab cluster layout and are kept as-is.
- Evaluated code base for the models: OpenClaw-RL `3573bc3e9a3fb73acdbb53bf83a6f72ca1ba7aea`; benchmark: harbor-framework/terminal-bench-2-1.
