# Multimodal RL Profiling Plan

## Goal
Measure distributed multimodal RL performance for `sync + colocate` and `async + disaggregate` GRPO pipelines, then attribute latency to rollout, reward, old log prob, actor update, queue wait, deserialization, and batch assembly.

## Benchmark Matrix
- Modes: `sync_colocate`, `fully_async_disaggregate`, `one_step_off_disaggregate`
- Model sizes: `3B`, `7B`, `32B`
- Scales: `1 node` and `2 node` runs on `4xGH200` nodes
- Default budget: `1 epoch`, `<64 GPUs`, small profiling datasets first

## Required Outputs
- `metrics.jsonl` via `trainer.logger=["console","file"]`
- `summary.json` and `summary.md` via `scripts/multimodality/summarize_profile.py`
- `nsys` traces for representative steps only
- Buffer analysis notes in `BUFFER_ANALYSIS.md`
