# Buffer Analysis

## What to Store
### Async queue / transfer queue
Prefer storing:
- prompt / response token ids
- rewards, advantages, masks, rollout parameter version
- media URI/path or compact sample identifiers
- image/video counts and optional cache keys

Avoid by default:
- raw image bytes in the queue
- full `pixel_values` tensors
- dense visual embeddings unless reuse clearly dominates transfer cost

### True replay pool
For a classic replay pool, default to:
- tokenized text trajectory data
- scalar training metadata
- media references rather than raw media tensors
- optional cached visual embeddings only when deterministic reuse is required and memory is budgeted

## Bottleneck Heuristic
Treat replay or queue storage as a bottleneck when any of the following is true:
- `timing_s/queue_wait / timing_s/step >= 0.15`
- queue payload remains large enough to limit concurrency or batch size
- stale sample counts grow with throughput
- trainer or rollouter idle ratios indicate producer/consumer mismatch
