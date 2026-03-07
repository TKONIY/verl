# Repository Guidelines

## Project Structure & Module Organization
`verl/` is the main Python package, split by runtime area such as `trainer/`, `workers/`, `models/`, `utils/`, and `experimental/`. Mirror new package code in `tests/` so coverage follows the same namespace shape. Use `examples/` for runnable recipes, `docs/` for Sphinx documentation, `scripts/` for maintenance helpers, `docker/` for container definitions, and `recipe/` for the submodule-managed recipes.

## Build, Test, and Development Commands
- `pip install -e .[test]` installs local contributor tooling.
- `pip install -e .[test,vllm]` or `pip install -e .[test,sglang]` adds rollout-specific extras.
- `pre-commit install && pre-commit run --all-files --show-diff-on-failure --color=always` runs the same checks used in CI.
- `pytest -s -x --asyncio-mode=auto tests/` runs the default local test suite.
- `cd docs && make html` builds the documentation site.
- `scripts/generate_trainer_config.sh` regenerates trainer config artifacts after config source changes.

## Cluster & Container Workflow
Use `podman-hpc` for containerized development in this environment; do not document plain `docker` commands as the default path. Prefer the ARM64 images and build scripts provided in `~/code/verl_docker`; use that image directly when available, or use the paired build script there to create a compatible ARM64 image before running experiments. Run GPU or distributed jobs only on allocated Slurm compute nodes, not login nodes. If your current shell is already on a compute node, treat it as an allocated worker, not as a launcher for nested local work: do not `srun` back onto the same allocation by default, and make sure any heavy `srun` launch explicitly requests newly allocated node resources instead of implicitly reusing the current node. Any heavyweight work such as image builds, package compilation, or large dependency installation must run on newly allocated compute nodes, not on login nodes and not by piggybacking on an existing interactive allocation. You are working on a large supercomputing cluster with many `4xGH200` nodes; keep runs under `64` GPUs total unless explicitly required, prefer the fewest epochs needed to answer the question, and never leave jobs running after the target evidence is collected. For interactive work, request resources first, for example `salloc -N 1 --gres=gpu:4 --cpus-per-task=32 --time=02:00:00`, then launch with `srun podman-hpc run ...`. For batch or multinode runs, prefer `sbatch` and start from `examples/slurm/ray_on_slurm.slurm`.

## Agent Tracking Documents
Maintain `agents/PROGRESS.md`, `agents/TODO.md`, and `agents/CHANGELOG.md` for every task. Update `agents/PROGRESS.md` with the active task and status, update `agents/TODO.md` with persistent follow-up items, and record repo-facing process or documentation changes in `agents/CHANGELOG.md`. Do not consider work complete until these documents reflect the latest state, the changes are committed, and the commit is pushed to `tkoniy`. Before each commit, review the working tree and clean out stale debug edits, abandoned script changes, and throwaway log files so only intentional code and reproducible artifacts remain staged.

## Coding Style & Naming Conventions
Target Python 3.10+ and follow Ruff defaults with a 120-character line limit. Use 4-space indentation, `snake_case` for modules and functions, `PascalCase` for classes, and test names beginning with `test_`. Prefer first-party imports under `verl`, and run `pre-commit` before opening a PR.

## Testing Guidelines
Pytest is the main framework, with `pytest-asyncio` for async code. CPU-only tests should end with `_on_cpu.py`; GPU-oriented tests use normal `test_*.py` names and are filtered by CI workflows. Keep routine tests in the matching namespace and reserve `tests/special_*` for distributed, e2e, or NPU-specific coverage. Update `.github/workflows/` when a new suite needs special hardware or exclusions.

## Commit & Pull Request Guidelines
Recent history follows a scoped subject format like `[trainer] feat: ...` or `[ckpt, model] fix: ...`; prepend `[BREAKING]` when needed. PR titles should match `[{modules}] {type}: {description}` where `{type}` is `feat`, `fix`, `refactor`, `chore`, or `test`. Include a concise summary, linked issues, test evidence, and docs or usage updates when behavior changes. After each completed local update, create a commit and push it to the `tkoniy` remote before considering the task done.
