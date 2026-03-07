# Results

## Expected Per-Run Record
- Mode and model size
- Node/GPU layout
- Mean `timing_s/step`
- Top 5 timing contributors
- `multimodal/payload_mb`
- `timing_s/queue_wait` and queue payload metrics for async runs
- Final judgment: whether buffer/replay is a primary, secondary, or non-primary bottleneck

## Recorded Runs
- `sync_colocate | 3B | 1 node x 4 GPU | run sync_colocate_3b_sglang_overlay_smoke4_20260307_020511`
  - Mean `timing_s/step`: `26.6535s`
  - Mean `timing_s/gen`: `16.6598s`
  - Mean `timing_s/old_log_prob`: `2.6153s`
  - Mean `timing_s/update_actor`: `1.5114s`
  - Mean `multimodal/payload_mb`: `23.0049`
  - Buffer verdict: `not the primary bottleneck`
- `sync_colocate | 7B | 1 node x 4 GPU | run sync_colocate_7b_sglang_20260307_020855`
  - Mean `timing_s/step`: `30.3625s`
  - Mean `timing_s/gen`: `16.6577s`
  - Mean `timing_s/old_log_prob`: `2.1026s`
  - Mean `timing_s/update_actor`: `1.8716s`
  - Mean `multimodal/payload_mb`: `27.0782`
  - Buffer verdict: `not the primary bottleneck`
- `sync_colocate | 32B | 1 node x 4 GPU | run sync_colocate_32b_sglang_2667913`
  - Status: `failed with OOM during checkpoint/model load on 4xGH200`
  - Observation: `single-node 4-GPU GH200 is not a stable configuration for this 32B VLM profile setup without further sharding or lower-memory rollout settings`
- `one_step_off_disaggregate | 3B | trainer 2 GPU + rollout 2 GPU | run one_step_off_disaggregate_3b_sglang_2667920`
  - Mean `timing_s/step`: `20.2142s`
  - Mean `timing_s/gen`: `11.5241s`
  - Mean `timing_s/ref`: `5.4495s`
  - Mean `timing_s/update_actor`: `1.6441s`
  - Mean `timing_s/sync_rollout_weights`: `0.8424s`
  - Mean `multimodal/payload_mb`: `18.5547`
  - Buffer verdict: `not the primary bottleneck`
- `one_step_off_disaggregate | 7B | trainer 2 GPU + rollout 2 GPU | run one_step_off_disaggregate_7b_sglang_2667925`
  - Mean `timing_s/step`: `32.2045s`
  - Mean `timing_s/gen`: `14.4204s`
  - Mean `timing_s/ref`: `12.6178s`
  - Mean `timing_s/update_actor`: `3.3193s`
  - Mean `timing_s/sync_rollout_weights`: `0.8664s`
  - Mean `multimodal/payload_mb`: `18.3752`
  - Buffer verdict: `not the primary bottleneck`
- `fully_async_disaggregate | 3B | trainer 2 GPU + rollout 2 GPU | run fully_async_disaggregate_3b_sglang_2667919`
  - Status: `blocked in current ARM64 image`
  - Observation: `after fixing CuDNN, hybrid_engine, and cupy, the rollout path still fails on missing module \`vllm\` from \`sglang.srt.weight_sync\``
  - Evidence: `runs/multimodality/slurm_logs/mm_fully_async_disaggregate_3b_sglang_2667919.out`
