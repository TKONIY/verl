# TODO

- Run `sync_colocate` on 7B with `1 epoch` and generate a first summary report.
- Run `fully_async_disaggregate` on 7B and compare queue wait, payload size, and stale-sample metrics.
- Run `one_step_off_disaggregate` as the lower-staleness async baseline.
- Validate 32B settings on GH200 without exceeding the `<64 GPU` rule.
- Compare replay storage strategies: metadata-only, preprocessed tensors, cached embeddings.
- Rebase the multimodal benchmark flow onto the `~/code/verl_docker` ARM64 image or its build script before the next fresh-node benchmark attempt.
- If the 3B `sync_colocate + sglang` smoke passes, scale to 4-GPU sync and then run the async/disaggregate variants on fresh nodes.
- Validate the persistent `~/code/verl_docker/runtime_overlays` CuDNN fix with a fresh-node `sync_colocate + 3B + sglang` smoke run, then fan out the benchmark matrix.

## Follow-up
- Add a `vllm`-capable ARM64 runtime path for `fully_async_disaggregate`.
- Retry `32B` with a more memory-efficient sharding or multinode layout.
