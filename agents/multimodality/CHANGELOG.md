# Changelog

## 2026-03-06
- Added `scripts/multimodality/run_profiled_grpo_vlm.sh` for sync and async multimodal GRPO profiling runs.
- Added `scripts/multimodality/summarize_profile.py` to generate timing and buffer reports from JSONL metrics.
- Added fully async queue instrumentation for wait, deserialize, assemble, and payload size.
- Added multimodal payload and visual token proxy metrics to PPO metric aggregation.
- Added `agents/multimodality/` tracking and planning documents.
- 2026-03-07: Documented the requirement to use `~/code/verl_docker` ARM64 images/build scripts and fresh compute allocations for heavyweight runtime preparation.
- 2026-03-07: Added `scripts/multimodality/run_profiled_grpo_vlm_apptainer.sh` and `scripts/multimodality/submit_profile_apptainer_slurm.sh` to run fresh-node ARM64 multimodal benchmarks with the `verl_docker` SIF path.
- 2026-03-07: Hardened the SIF launcher to bind HF cache at `/cache/huggingface`, localize HF model snapshots, tee `train.log`, and keep `RAY_TMPDIR` short enough for Ray socket limits.
- 2026-03-07: Changed the ARM64 apptainer launcher to repair SGLang CuDNN via a persistent overlay under `~/code/verl_docker/runtime_overlays`, keeping heavyweight package writes off the container tmpfs.
- 2026-03-07: Bound the shared `~/code/verl_docker/runtime_overlays` CuDNN/CUBLAS directories onto the container `site-packages` tree so fresh-node Ray/SGLang workers see CuDNN 9.16 without source edits.
- 2026-03-07: Added `agents/multimodality/FINAL_REPORT.md` and `runs/multimodality/summary_suite.md` summarizing the successful runs, the `fully_async` dependency blocker, and the `32B` OOM boundary.
