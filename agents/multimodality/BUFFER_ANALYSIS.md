# Buffer Analysis

## What to Store
### Async queue / transfer queue
Prefer storing:
- prompt / response token ids
- scalar rewards, advantages, masks, action log-probs, rollout parameter version
- media URI/path or compact sample identifiers
- image/video counts and optional cache keys
- for VLA, proprio/state tensors, action chunks, done flags, and simulator step metadata

Avoid by default:
- raw image bytes in the queue
- full `pixel_values` tensors
- dense visual embeddings unless reuse clearly dominates transfer cost
- serialized simulator objects or large uncompressed frame stacks

### True replay pool
For a replay pool, default to:
- tokenized text trajectory data
- scalar training metadata
- media references rather than raw media tensors
- VLA observation references plus compact low-dimensional state
- optional cached visual embeddings only when deterministic reuse is required and memory is budgeted

## Bottleneck Heuristic
Treat replay or queue storage as a bottleneck when any of the following is true:
- `timing_s/queue_wait / timing_s/step >= 0.15`
- queue payload remains large enough to limit concurrency or batch size
- stale sample counts grow with throughput
- trainer or rollouter idle ratios indicate producer/consumer mismatch
- VLA simulator or image decode time begins exceeding actor-update time

## Current Verdict
- In the successful multimodal runs, replay storage is **not** the dominant bottleneck.
- The dominant costs are rollout generation, reference / old-log-prob recomputation, and weight synchronization.
- At `32 GPU`, payload size grows sharply (`84.86 MB` sync vs `207.37 MB` one-step disaggregate), so raw-media storage would likely become a bottleneck at the next scale jump.
- For VLA, the likely first bottlenecks are simulator / environment latency and observation serialization, not the replay schema itself.
