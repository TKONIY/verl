# Multimodal RL Distributed Performance Report

## Scope
- Cluster path: fresh Slurm compute nodes only
- Runtime path: `~/code/verl_docker/verl_sgl056_arm64_latest.sif` with persistent overlay under `~/code/verl_docker/runtime_overlays/sglang_cudnn916_py310_arm64`
- Modes attempted: `sync_colocate`, `one_step_off_disaggregate`, `fully_async_disaggregate`
- Model sizes attempted: `3B`, `7B`, `32B`

## Repro Scripts
- Base launcher: `scripts/multimodality/run_profiled_grpo_vlm.sh`
- ARM64 container launcher: `scripts/multimodality/run_profiled_grpo_vlm_apptainer.sh`
- Fresh-node submit helper: `scripts/multimodality/submit_profile_apptainer_slurm.sh`
- Aggregation helpers: `scripts/multimodality/summarize_profile.py`, `scripts/multimodality/summarize_suite.py`
- Successful run artifacts:
  - `runs/multimodality/sync_colocate_3b_sglang_overlay_smoke4_20260307_020511`
  - `runs/multimodality/sync_colocate_7b_sglang_20260307_020855`
  - `runs/multimodality/one_step_off_disaggregate_3b_sglang_2667920`
  - `runs/multimodality/one_step_off_disaggregate_7b_sglang_2667925`

## Successful Measurements
- `sync_colocate 3B`: step `26.65s`, gen `16.66s`, old-log-prob `2.62s`, actor update `1.51s`, payload `23.00 MB`
- `sync_colocate 7B`: step `30.36s`, gen `16.66s`, ref `6.29s`, actor update `1.87s`, payload `27.08 MB`
- `one_step_off 3B`: step `20.21s`, gen `11.52s`, ref `5.45s`, actor update `1.64s`, sync-rollout-weights `0.84s`, payload `18.55 MB`
- `one_step_off 7B`: step `32.20s`, gen `14.42s`, ref `12.62s`, actor update `3.32s`, sync-rollout-weights `0.87s`, payload `18.38 MB`

## Findings
- Generation dominates end-to-end step time in every successful run.
- `one_step_off_disaggregate` improves `3B` step time versus `sync_colocate` (`20.21s` vs `26.65s`) by overlapping rollout and update work.
- At `7B`, async disaggregation does not beat sync on total step time (`32.20s` vs `30.36s`) because reference / sync-update cost grows materially.
- Multimodal payloads stay in an `18-27 MB` range per sampled batch in these short runs; this is noticeable but not yet the primary bottleneck.
- Replay / queue recommendation: store tokenized text trajectory data, scalar rewards / advantages / masks, rollout parameter version, and media references (`image path`, `video path`, cache key). Do **not** store raw image bytes or dense vision tensors in the replay / transfer path by default.

## Bottleneck Assessment
- With current evidence, multimodal replay storage is **not** the primary bottleneck for the successful runs.
- The most sensitive components are generation, reference / old-log-prob recomputation, and weight-sync overhead in async disaggregate mode.
- Replay becomes a likely bottleneck only if queue wait starts consuming >15% of step time, or if raw media tensors are pushed through the queue.

## Failed / Boundary Cases
- `fully_async_disaggregate 3B`: environment and config fixes succeeded through CuDNN, hybrid-engine, and `cupy`, but the rollout path still failed on missing `vllm` import inside `sglang.srt.weight_sync`; see `runs/multimodality/slurm_logs/mm_fully_async_disaggregate_3b_sglang_2667919.out`.
- `sync_colocate 32B`: failed with OOM during model load on `1 node x 4 GPU`; see `runs/multimodality/slurm_logs/mm_sync_colocate_32b_sglang_2667913.err`.

## Practical Conclusion
- For this ARM64 GH200 environment, the most reliable distributed multimodal RL configuration today is `one_step_off_disaggregate` for `3B`, with `7B` still feasible but no longer clearly faster than sync.
- `sync_colocate` remains the cleanest baseline and scales from `3B` to `7B` without major surgery.
- `fully_async_disaggregate` needs an image that also includes a working `vllm` dependency chain before it can be profiled fairly on this cluster.
