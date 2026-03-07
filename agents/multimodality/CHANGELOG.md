# Changelog

## 2026-03-06
- Added multimodal profiling scaffolding and planning documents.

## 2026-03-07
- Added `agents/multimodality/MAIN_WORKTREE_HANDOFF.md`, `agents/multimodality/VLA_LOW_MEMORY_PLAN.md`, and `runs/multimodality/component_breakdown.md` for integration and follow-up execution.
- Added `scripts/multimodality/extract_trainlog_metrics.py` to recover final metrics from `train.log`.
- Added `scripts/multimodality/submit_profile_apptainer_ray_slurm.sh` for fresh-node Ray-on-Slurm distributed runs.
- Generated summary JSON files under `runs/multimodality/summaries/` and a suite table at `runs/multimodality/summary_suite.md`.
- Added VLA runtime helpers under `scripts/vla/` and kept environment fixes in script/runtime-overlay space instead of modifying core source.
- Updated the VLA Apptainer launcher to keep Ray temp files off read-only `/local`.
