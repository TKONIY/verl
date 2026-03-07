# TODO

## Process Maintenance
- Keep `agents/PROGRESS.md` updated with the current task status.
- Keep `agents/CHANGELOG.md` updated with every repo-facing process or documentation change.
- Review `AGENTS.md` whenever contributor workflow requirements change.
- Do not treat work as complete until updates are committed and pushed to the `tkoniy` remote.

## Multimodality Benchmarks
- Run and summarize `sync_colocate`, `fully_async_disaggregate`, and `one_step_off_disaggregate` VLM profiling jobs.
- Validate queue wait, payload size, and stale-sample behavior on 7B before expanding 32B runs.
- Keep default benchmark runs short and shut down jobs immediately after collecting the target traces.
- Keep future multimodal experiment launchers on fresh Slurm allocations when operating from an existing compute node session.
- Switch multimodal launchers to the `~/code/verl_docker` ARM64 image path or its build script before running the remaining benchmarks.
- Reuse the persistent `~/code/verl_docker/runtime_overlays` CuDNN overlay for the remaining fresh-node SGLang benchmark jobs.
- Before the final commit, prune stale modified-code experiments and non-essential logs from the working tree so the pushed change set stays focused.

## Follow-up
- Build or obtain an ARM64 image that includes a working `vllm` dependency chain so `fully_async_disaggregate` can be profiled end-to-end.
- Revisit `32B` with lower-memory rollout settings, more shards, or multinode placement before treating it as supported on `1 x 4 GH200`.
