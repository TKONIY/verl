# Progress

## Current Status
- Multimodal GRPO profiling is complete for the requested distributed comparison matrix that was runnable in this environment.
- Successful measurements exist for `sync_colocate` and `one_step_off_disaggregate` at both `1 node` and `32 GPU / 8 node` scales.
- `fully_async_disaggregate + sglang` now initializes correctly but stalls before the first optimization step in this ARM64 GH200 environment.
- VLA Libero profiling scripts are in place; the latest smoke runs clear dependency import and `ray.init`, but still fail before the first training step due high-memory worker materialization in this environment.

## Latest Update
- Date: 2026-03-07
- Task: Consolidate multimodal and VLA profiling evidence into final reproducible reports.
- State: Reports updated; VLA blocker documented with final evidence.
- 2026-03-07: Saved extracted metric JSON files under `runs/multimodality/summaries/` for all successful baseline runs.
- 2026-03-07: Added `runs/multimodality/summary_suite.md` with a single comparison table covering `1 node` and `32 GPU` runs.
- 2026-03-07: Confirmed the `fully_async` terminal symptom is a pre-step stall after `CheckpointEngineWorker` initialization, not the earlier missing-`vllm` import path.

## Final Status
- Multimodal profiling report is ready.
- Only cleanup, commit, and push remain.
