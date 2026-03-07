# Multimodal and VLA Distributed Profiling Report

## Scope
- Cluster policy: fresh Slurm compute nodes only, never login nodes for GPU work.
- Runtime path: `~/code/verl_docker/verl_sgl056_arm64_latest.sif` plus shared overlays under `~/code/verl_docker/runtime_overlays/`.
- Multimodal modes: `sync_colocate`, `one_step_off_disaggregate`, `fully_async_disaggregate`.
- Multimodal sizes: `3B`, `7B`, `32B` boundary check, plus a `32 GPU` scale comparison.
- VLA path: Libero GRPO smoke profiling with the same container workflow.

## Reproducibility Assets
- Multimodal launcher: `scripts/multimodality/run_profiled_grpo_vlm.sh`
- Multimodal Apptainer launcher: `scripts/multimodality/run_profiled_grpo_vlm_apptainer.sh`
- Multinode Ray-on-Slurm helper: `scripts/multimodality/submit_profile_apptainer_ray_slurm.sh`
- Train-log metric extraction: `scripts/multimodality/extract_trainlog_metrics.py`
- Metric JSON outputs: `runs/multimodality/summaries/*.json`
- Suite comparison table: `runs/multimodality/summary_suite.md`
- VLA launcher: `scripts/vla/run_profiled_libero_grpo_apptainer.sh`
- VLA Slurm helper: `scripts/vla/submit_profile_libero_grpo_slurm.sh`
- VLA runtime prep: `scripts/vla/setup_vla_runtime_overlay.sh`

## Pipeline Coordination
```text
        +--------------------+        samples / prompts / media refs
        |   Dataset Loader   | -----------------------------------------+
        +--------------------+                                          |
                                                                       v
+------------------+     token ids / image refs / rollout version   +-------------------+
| Trainer / Actor  | <--------------------------------------------> | Rollout Workers   |
| PPO / GRPO step  |      rewards / logprobs / masks / seq lens     | sglang / envloop  |
+------------------+                                                 +-------------------+
        |                                                                          |
        | gradients / updated weights                                              | generated responses / env transitions
        v                                                                          v
+------------------+     checkpoints / parameter sync / queues      +-------------------+
| Checkpoint / Sync| <--------------------------------------------> | Replay / Buffers   |
+------------------+         metadata, cache keys, compact state    +-------------------+
```

## What Moves Between Stages
- Text side: prompt ids, response ids, attention masks, position ids, rewards, advantages, KL metadata, rollout parameter version.
- Vision side: image/video references, counts, optional cache keys, and compact derived stats such as visual token proxy.
- Async-only path: queue payload metadata, sample freshness / staleness signals, and rollout-weight synchronization events.
- VLA-specific path: low-dimensional robot state, action chunks, done flags, and observation references.

## Multimodal Results
### 1-node baselines
- `sync_colocate 3B`: step `26.65s`, gen `16.66s`, payload `23.00 MB`, throughput `46.97`
- `sync_colocate 7B`: step `30.36s`, gen `16.66s`, payload `27.08 MB`, throughput `41.64`
- `one_step_off_disaggregate 3B`: step `20.21s`, gen `11.52s`, payload `18.55 MB`, throughput `104.85`
- `one_step_off_disaggregate 7B`: step `32.20s`, gen `14.42s`, payload `18.38 MB`, throughput `80.33`

### 32-GPU comparison
- `sync_colocate 3B | 8 node / 32 GPU`: step `27.57s`, gen `6.32s`, old-log-prob `4.57s`, update-weights `8.75s`, payload `84.86 MB`, throughput `19.20`
- `one_step_off_disaggregate 3B | 8 node / 32 GPU`: step `52.17s`, gen `13.07s`, sync-rollout-weights `12.73s`, update-weights `25.17s`, payload `207.37 MB`, throughput `44.21`

## Interpretation
- Generation is still the largest single component in every successful run.
- `one_step_off_disaggregate` is clearly best at `3B` on `1 node`, cutting step time from `26.65s` to `20.21s`.
- At `7B`, async disaggregation loses its edge because reference and sync costs grow faster than the rollout overlap benefit.
- At `32 GPU`, `one_step_off_disaggregate` raises throughput but also expands payload size and weight-sync overhead sharply; that mode is no longer latency-efficient per step.
- The `32 GPU` sync baseline keeps per-step latency close to the `1 node` case because generation parallelizes well, but total distributed coordination still limits end-to-end throughput efficiency.

## Timing and Transfer Share
- Quantitative component shares are tabulated in `runs/multimodality/component_breakdown.md`.
- For the most relevant `3B` runs:
  - `sync | 1 node`: generation `62.5%`, transport proxy `10.8%`, overlap lower bound `0.0%`
  - `one_step_off | 1 node`: generation `57.0%`, transport proxy `12.0%`, overlap lower bound `4.1%`
  - `sync | 32 GPU`: generation `22.9%`, transport proxy `31.7%`, overlap lower bound `0.0%`
  - `one_step_off | 32 GPU`: generation `25.1%`, transport proxy `72.6%`, overlap lower bound `23.6%`
- Interpretation:
  - On `1 node`, multimodal transfer / synchronization is noticeable but still secondary to generation and reference computation.
  - On `32 GPU`, transport and synchronization become first-order costs, especially for `one_step_off_disaggregate` where weight movement and synchronization dominate the step timeline.
  - The nonzero overlap lower bound in `one_step_off` means some communication is hidden under compute, but not enough to offset the large transport growth at scale.

## Cross-Machine Multimodal Transfer
- Across machines, the system does **not** literally move only metadata end-to-end. What should stay as metadata in replay is the persistent representation: media path, cache key, and compact identifiers.
- During execution, each stage still must receive enough materialized data to do work:
  - prompt / response token ids
  - masks and scalar PPO metadata
  - rollout parameter version
  - either decoded multimodal tensors, or references that trigger remote fetch / decode on the consumer side
- In this cluster setup, shared storage lets us avoid serializing raw image bytes through the replay queue itself, but that does **not** remove cross-machine data motion; it shifts the cost into worker-side fetch / decode, object-store movement, and synchronization.
- Therefore `multimodal/payload_mb` should be read as the effective multimodal batch payload associated with a training step, while `update_weights + sync_rollout_weights` is the best available proxy for transport / synchronization time in the current logs.

## Replay / Buffer Conclusion
- The replay path should store tokenized trajectories, scalar learning metadata, rollout versioning, and media references.
- Do not default to raw image bytes, decoded `pixel_values`, dense visual embeddings, or simulator objects inside replay.
- In the successful multimodal runs, replay is **not** the dominant bottleneck; rollout generation and synchronization are.
- For VLA, the more likely near-term bottlenecks are simulator / observation serialization and env-loop latency, not replay layout.

## Failed or Partial Cases
- `fully_async_disaggregate + sglang`: now gets through initialization, but still stalls before the first training step; latest evidence is `runs/multimodality/fully_async_disaggregate_3b_sglang_smoke3_wt_20260307_113040/train.log`.
- `sync_colocate 32B`: OOM on `1 x 4 GH200`; a larger-shard or multinode layout is required.

## VLA Status
- The VLA Libero path now passes dependency import checks inside the container, reaches trainer startup, and successfully initializes local Ray.
- Earlier blockers fixed without source edits: missing `compute_reward` shim, LIBERO source availability, `robosuite` macro import, `termcolor`, MuJoCo EGL configuration, `/local` temp-dir writes, and Ray temp-dir placement.
- The remaining blocker is pre-step worker / model materialization pressure: `vla_libero_grpo_smoke10_wt_20260307_231332` hits Slurm OOM shortly after local Ray startup, while `vla_libero_grpo_smoke11_wt_20260307_231610` avoids immediate OOM but shows no post-`ray.init` progress and was cancelled after ~5 minutes at roughly `110 GB` RSS to avoid burning GPUs.
- Conclusion for this task: the VLA example is launchable and debuggable with the provided scripts, but a first-step-stable profile still needs either more host memory headroom or a lower-replication VLA runtime plan.

## Bottom Line
- Best latency-oriented multimodal configuration here: `one_step_off_disaggregate | 3B | 1 node`.
- Best large-scale stability baseline: `sync_colocate | 3B | 32 GPU`.
- Most important systems conclusion: at larger distributed scale, synchronization and payload growth dominate before replay storage itself becomes the primary bottleneck.
