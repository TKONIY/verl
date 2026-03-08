# Progress

## Current Status
- Using the isolated shared-filesystem worktree at `../verl-codex-mm` to avoid overlapping another Codex agent's edits in the main worktree.
- Completed multimodal distributed RL profiling scripts, Slurm launchers, train-log metric extraction, and reproducible result summaries.
- Verified successful multimodal runs for `sync_colocate` and `one_step_off_disaggregate` on `1 node` and `32 GPU / 8 node` scales.
- Brought the VLA Libero path past dependency/import failures and onto a runnable Slurm+Apptainer path; current focus is the final end-to-end smoke validation.

## Latest Update
- Date: 2026-03-07
- Task: Finish multimodal + VLA profiling artifacts, update reports, and prepare the final push to `tkoniy`.
- State: Final validation and report consolidation in progress.
- 2026-03-07: Created an isolated shared-filesystem worktree so report and script changes do not conflict with parallel edits in the main repo.
- 2026-03-07: Added `scripts/multimodality/extract_trainlog_metrics.py` and generated reproducible JSON summaries under `runs/multimodality/summaries/`.
- 2026-03-07: Confirmed the large-scale comparison pair: `sync_colocate | 3B | 32 GPU` and `one_step_off_disaggregate | 3B | 32 GPU`.
- 2026-03-07: Updated the VLA launcher to fix `DISAGG_SIM_ENABLE` expansion and to force Ray temp directories onto writable container paths instead of `/local`.
- 2026-03-07: Added a quantitative component/transport-share breakdown plus a main-worktree cherry-pick handoff and a lower-memory VLA follow-up plan.
- 2026-03-07: Added a concise image-transfer summary explaining when images are fetched, when only metadata should move, and why cross-machine cost is often shared-storage fetch plus synchronization rather than raw-byte RPC alone.
- 2026-03-08: Added an external `verl-agent` inspection note and an agentic VLM open-source comparison to guide follow-on benchmark and tooling work.

## Final Status
- Multimodal profiling is complete with reproducible scripts, extracted metrics, suite summaries, and a final report.
- VLA smoke validation reached dependency import and `ray.init`, but the final 1-step Libero smoke still does not complete: one run hit Slurm OOM after local Ray startup and a lower-footprint rerun stalled at ~110 GB RSS before first-step logs, so the final blocker is documented rather than left running.
