# TODO

## Process Maintenance
- Keep `agents/PROGRESS.md`, `agents/TODO.md`, and `agents/CHANGELOG.md` aligned with the current task before commit.
- Clean stale debug logs, abandoned temp outputs, and throwaway modified files before the final push.
- Do not treat the task as done until the final commit is pushed to `tkoniy`.

## Multimodal / VLA Follow-up
- If `fully_async + sglang` is revisited, debug the control-flow stall before first step rather than retrying more container permutations.
- Revisit `32B` only with a more memory-efficient sharding or multinode model placement plan.
- Expand VLA profiling from the Libero smoke path to a larger matrix only after the 1-step smoke path remains stable.
