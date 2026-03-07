# Progress

## Current Status
- Created the `agents/` workspace for contributor-tracking documents.
- Added baseline process documentation for progress, todo tracking, and changelog updates.
- Updated `AGENTS.md` so each task must refresh these tracking files, commit changes, and push them to `tkoniy`.
- Added multimodal RL benchmark, profiler, and reporting scaffolding under `scripts/multimodality/` and `agents/multimodality/`.

## Latest Update
- Date: 2026-03-06
- Task: Implement multimodal distributed RL profiling scaffolding, replay/buffer analysis docs, and reproducible experiment scripts.
- State: Implemented, validating and summarizing
- 2026-03-07: Added Slurm guidance that when already on a compute node, heavy `srun` jobs must request fresh node allocations rather than reusing the current node.
- 2026-03-07: Added ARM64 container guidance to use `~/code/verl_docker` images or build scripts, and require fresh compute-node allocations for heavyweight build and compile tasks.
- 2026-03-07: Switched the ARM64 SIF launcher from in-container CuDNN writes to a persistent `~/code/verl_docker/runtime_overlays` overlay on shared storage so fresh-node jobs can repair SGLang CuDNN without exhausting writable tmpfs.
- 2026-03-07: Confirmed heavyweight runtime fixes must stay on fresh compute nodes and updated the ARM64 SIF launcher to bind the persistent CuDNN/CUBLAS overlay directly onto container `site-packages` paths.
- 2026-03-07: Added a pre-commit cleanup rule to `AGENTS.md` requiring a pass over modified code and logs so stale debug changes and throwaway log files are removed before pushing to `tkoniy`.
- 2026-03-07: Updated the fully-async launcher path to pass `actor_rollout_ref.hybrid_engine=False`, matching the documented async shell examples and avoiding the trainer-side assertion.
- 2026-03-07: Extended the persistent runtime overlay to carry both CuDNN and `cupy-cuda12x`, and propagated the overlay through Ray via `ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH`.
- 2026-03-07: Updated the one-step-off launcher path to pass `actor_rollout_ref.hybrid_engine=False`, matching the async/disaggregate requirement without touching source code.

## Final Status
- 2026-03-07: Completed the multimodal distributed RL profiling pass on fresh compute nodes, with successful `sync_colocate` runs at `3B` and `7B`, successful `one_step_off_disaggregate` runs at `3B` and `7B`, a documented `32B` single-node OOM boundary, and a documented `fully_async_disaggregate` blocker in the current ARM64 image.
