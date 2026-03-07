# Changelog

## 2026-03-06
- Added the `agents/` directory for operational tracking documents.
- Added `agents/PROGRESS.md`, `agents/TODO.md`, and `agents/CHANGELOG.md`.
- Updated `AGENTS.md` to require maintaining tracking docs, committing changes, and pushing to `tkoniy`.

## 2026-03-07
- Added a main-worktree cherry-pick handoff note, a lower-memory VLA follow-up plan, and a quantitative multimodal component/transport-share breakdown.
- Added multimodal profiling scripts and Slurm launchers for fresh-node ARM64 runs.
- Added train-log metric extraction so successful runs without `metrics.jsonl` still produce reproducible summaries.
- Added a shared-filesystem isolated worktree workflow to avoid collisions with concurrent agent edits.
- Added VLA Libero Slurm launchers and runtime-overlay setup scripts without modifying core repo source for environment fixes.
- Updated the VLA launcher to pin Ray temp directories to writable paths inside the container.
