# Progress

## Current Status
- Added a reusable launcher for profiled multimodal GRPO runs.
- Added structured metric summarization for timing, multimodal payload, and async buffer metrics.
- Added queue-level timing hooks for fully async training.

## Latest Update
- Date: 2026-03-06
- Task: Implement multimodal benchmark/profiler scaffolding and analysis docs.
- State: Implemented, pending validation summary
- 2026-03-07: Environment guidance updated to prefer `~/code/verl_docker` ARM64 images and to reserve fresh compute nodes for heavyweight build/install steps.
- 2026-03-07: Verified `~/code/verl_docker/verl_sgl056_arm64_latest.sif` works with the current bound worktree and launched the first fresh-node `sync_colocate + 3B + sglang` smoke run.
- 2026-03-07: Isolated and fixed two SIF launcher issues: invalid container HF cache path and Ray AF_UNIX socket overlong paths from an overly long `RAY_TMPDIR`.
- 2026-03-07: Reworked the SIF launcher to mount `~/code/verl_docker` and install the required CuDNN wheel into a persistent shared overlay, avoiding writable-tmpfs exhaustion on fresh compute nodes.
- 2026-03-07: Confirmed the persistent overlay exposes CuDNN through `importlib.metadata`; updated the launcher probe accordingly after the first fresh-node overlay install.
- 2026-03-07: Capped actor and log-prob micro-batches by the normalized PPO mini-batch per trainer GPU so short fresh-node smoke runs stay script-only and avoid FSDP batch-shape assertions.
- 2026-03-07: Replaced the failed `LD_LIBRARY_PATH`-only approach with direct bind mounts of the persistent overlay onto container `site-packages/nvidia/{cudnn,cublas}`, while keeping all heavyweight verification on newly allocated compute nodes.
- 2026-03-07: Tightened `summarize_profile.py` to exclude non-time helper fields like prompt and response lengths from the ranked timing table.
- 2026-03-07: Diagnosed the first `fully_async` failure as a config issue, then aligned the launcher with official async shell examples by forcing `actor_rollout_ref.hybrid_engine=False`.
- 2026-03-07: Added `cupy-cuda12x` staging to the shared overlay and propagated it to Ray workers after the first fully-async failure showed NCCL checkpoint registration was blocked by a missing `cupy` import.
- 2026-03-07: Confirmed `one_step_off_disaggregate` shares the same hybrid-engine assertion path as `fully_async` and updated the launcher override accordingly.

## Final Status
- 2026-03-07: Finished the planned fresh-node benchmark pass and published summaries for `sync_colocate` (`3B`, `7B`) and `one_step_off_disaggregate` (`3B`, `7B`), with explicit blocker notes for `fully_async_disaggregate` and `32B` single-node memory limits.
