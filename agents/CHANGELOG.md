# Changelog

## 2026-03-06
- Added the `agents/` directory for operational tracking documents.
- Added `agents/PROGRESS.md` for current status tracking.
- Added `agents/TODO.md` for persistent workflow tasks.
- Added `agents/CHANGELOG.md` for process-level change history.
- Updated `AGENTS.md` to require maintaining the `agents/` tracking documents on every task.
- Updated `AGENTS.md` to require committing completed local changes and pushing them to the `tkoniy` remote.
- Clarified that task completion requires both committing and pushing, not just creating a local commit.
- Added GH200 cluster guidance: stay below 64 GPUs, prefer short runs, and do not leave jobs running after data collection.
- Added multimodal RL profiling scripts, reports, and task docs for sync/async distributed experiments.
- 2026-03-07: Clarified in `AGENTS.md` that compute-node sessions must launch heavy `srun` work on newly allocated nodes, not implicitly on the current node.
- 2026-03-07: Clarified in `AGENTS.md` that ARM64 runs should use `~/code/verl_docker` images/build scripts and that heavyweight container/build work must execute on newly allocated compute nodes.
- 2026-03-07: Updated `scripts/multimodality/run_profiled_grpo_vlm_apptainer.sh` to stage the required `nvidia-cudnn-cu12==9.16.0.29` into a persistent overlay under `~/code/verl_docker/runtime_overlays` instead of writing into the container tmpfs.
- 2026-03-07: Updated the ARM64 SIF launcher to bind the persistent `~/code/verl_docker/runtime_overlays` CuDNN/CUBLAS tree onto container `site-packages/nvidia/{cudnn,cublas}` so Ray and SGLang inherit CuDNN 9.16 on fresh compute nodes.
- 2026-03-07: Clarified in `AGENTS.md` that every commit should include a cleanup pass over the modified code and log files, removing stale scratch changes before push.
- 2026-03-07: Patched `scripts/multimodality/run_profiled_grpo_vlm.sh` so `fully_async_disaggregate` launches with `actor_rollout_ref.hybrid_engine=False` via script overrides instead of source edits.
- 2026-03-07: Rewrote the ARM64 apptainer launcher so the shared runtime overlay can also install `cupy-cuda12x` for fully-async runs while still binding CuDNN/CUBLAS into container site-packages.
- 2026-03-07: Patched `scripts/multimodality/run_profiled_grpo_vlm.sh` so `one_step_off_disaggregate` also disables `actor_rollout_ref.hybrid_engine` through script overrides.
