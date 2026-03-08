# Progress

## Current Status
- Created the `agents/` workspace for contributor-tracking documents.
- Added baseline process documentation for progress, todo tracking, and changelog updates.
- Updated `AGENTS.md` so each task must refresh these tracking files, commit changes, and push them to `tkoniy`.
- Added multimodal RL benchmark, profiler, and reporting scaffolding under `scripts/multimodality/` and `agents/multimodality/`.

## Latest Update
- Date: 2026-03-08
- Task: Validate `cosmos-rl` runtime feasibility, compare it with `verl`, and land a reproducible smoke-run path plus bilingual docs.
- State: Completed `cosmos-rl` smoke execution and architecture analysis; documenting reproducible setup and remaining blockers.
- 2026-03-07: Added `agents/wm/VLA_RL_MODES.md` under `agents/wm/` with a Chinese quick-reference note covering repository task types and a `PPO / GRPO / SAC / SFT` training-mode comparison.
- 2026-03-07: Added `agents/wm/VLA_RL_MODES_ONEPAGE.md` as a compact one-page cheat sheet for faster onboarding and presentation use.
- 2026-03-07: Added Slurm guidance that when already on a compute node, heavy `srun` jobs must request fresh node allocations rather than reusing the current node.
- 2026-03-07: Added ARM64 container guidance to use `~/code/verl_docker` images or build scripts, and require fresh compute-node allocations for heavyweight build and compile tasks.
- 2026-03-07: Switched the ARM64 SIF launcher from in-container CuDNN writes to a persistent `~/code/verl_docker/runtime_overlays` overlay on shared storage so fresh-node jobs can repair SGLang CuDNN without exhausting writable tmpfs.
- 2026-03-07: Confirmed heavyweight runtime fixes must stay on fresh compute nodes and updated the ARM64 SIF launcher to bind the persistent CuDNN/CUBLAS overlay directly onto container `site-packages` paths.
- 2026-03-07: Added a pre-commit cleanup rule to `AGENTS.md` requiring a pass over modified code and logs so stale debug changes and throwaway log files are removed before pushing to `tkoniy`.
- 2026-03-07: Updated the fully-async launcher path to pass `actor_rollout_ref.hybrid_engine=False`, matching the documented async shell examples and avoiding the trainer-side assertion.
- 2026-03-07: Extended the persistent runtime overlay to carry both CuDNN and `cupy-cuda12x`, and propagated the overlay through Ray via `ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH`.
- 2026-03-07: Updated the one-step-off launcher path to pass `actor_rollout_ref.hybrid_engine=False`, matching the async/disaggregate requirement without touching source code.

- 2026-03-08: Added `scripts/run_cosmos_rl_smoke_apptainer.sh` to reproduce a `1 GPU` `cosmos-rl` SFT smoke run inside the ARM64 apptainer image with lightweight compatibility wrappers.
- 2026-03-08: Ran `cosmos-rl` successfully on host `nid010055` with `1 x NVIDIA GH200 120GB`, completing `Step: 1/1` on `Qwen/Qwen2.5-0.5B-Instruct` and the local `sharegpt52k_small` fixture dataset.
- 2026-03-08: Added bilingual `docs/examples/cosmos_rl_vs_verl.md` and `docs/examples/cosmos_rl_vs_verl_zh.md` describing direct-run feasibility, `cosmos-rl` vs `verl` architecture, and the practical compatibility gaps in the current environment.
## Final Status
- 2026-03-07: Implemented root-level third-party world-model intake plus experimental `CosmosEnv` support, with multi-GPU launchers and repo-role analysis docs for external VLA/world-model projects.
- 2026-03-07: Completed the multimodal distributed RL profiling pass on fresh compute nodes, with successful `sync_colocate` runs at `3B` and `7B`, successful `one_step_off_disaggregate` runs at `3B` and `7B`, a documented `32B` single-node OOM boundary, and a documented `fully_async_disaggregate` blocker in the current ARM64 image.
- 2026-03-07: Added a true `apptainer + ray_on_slurm` submit path for multinode runs, replacing the invalid `-N 2 --ntasks=1` shortcut that Slurm collapsed back to one node.
- 2026-03-07: Fixed the true multinode Ray launcher to force `RAY_TMPDIR=/tmp/ray` and `ray start --temp-dir /tmp/ray` after the first 32-GPU run failed on a read-only `/local` temp path.
- 2026-03-07: Added root-level `third_party/` submodules for `cosmos-predict2.5`, `cosmos-transfer2.5`, `cosmos-reason2`, `cosmos-rl`, `lingbot-vla`, `lingbot-va`, `dreamzero`, and `Motus`.
- 2026-03-07: Added experimental `simulator_type=cosmos` support to `verl.experimental.vla` with a new `CosmosEnv`, a smoke-testable `mock` backend, and a future-facing `predict2` adapter path.
- 2026-03-07: Added `prepare_cosmos_dataset.py`, `run_pi05_cosmos_sac.sh`, and `run_pi05_cosmos_sac_disagg.sh` for single-node and disaggregated multi-GPU world-model RL demos.
- 2026-03-07: Added world-model/VLA role analysis docs under `agents/wm/` and public notes at `docs/examples/cosmos_world_model_rl.md` and `docs/examples/cosmos_world_model_rl_zh.md` covering `Cosmos`, `LingBot`, `DreamZero`, and `Motus`.
- 2026-03-07: Added English and Chinese design notes for a dedicated world-model / embodied robotics RL framework, highlighting how role-composable world-action models differ from the current `verl` abstraction.
- 2026-03-08: Validated that `cosmos-rl` is not bare-host runnable here, but is runnable in the ARM64 apptainer image after adding missing Python packages plus compatibility shims for `redis-server` and `torchrun`.
