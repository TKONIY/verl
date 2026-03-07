# Installation and repository intake

## Scope

This note documents the installation and intake convention for the new root-level `third_party/` tree.

## Repositories

The following repositories are added as git submodules:

- `third_party/cosmos-predict2.5`
- `third_party/cosmos-transfer2.5`
- `third_party/cosmos-reason2`
- `third_party/cosmos-rl`
- `third_party/lingbot-vla`
- `third_party/lingbot-va`
- `third_party/dreamzero`
- `third_party/Motus`

## Initialization

Clone or refresh submodules with:

```bash
git submodule update --init --recursive third_party
```

If a single submodule needs to be refreshed:

```bash
git submodule update --init --recursive third_party/cosmos-predict2.5
```

## Why this lives at the repo root

This repository already contains `verl/third_party/` for import-compatibility shims and vendored runtime patches. The new `third_party/` directory is intentionally separate:

- `verl/third_party/` remains Python-package-facing internal glue.
- `third_party/` stores upstream external research repositories with minimal reshaping.

This separation keeps upstream sync simpler and avoids mixing import shims with full external projects.

## Cosmos-specific note

The current `verl.experimental.vla` integration uses `simulator_type=cosmos` with `env.train.cosmos.backend=mock` by default for smoke tests.

The code also validates a future `cosmos_predict2`-style path through `third_party/cosmos-predict2.5`, but the official robot action-conditioned inference path is still file-oriented and documented as single-GPU, so the current patch does not claim full production step-wise Cosmos inference yet.

## Recommended setup order

1. Initialize `third_party/` submodules.
2. Generate a minimal dataset with `python -m verl.experimental.vla.prepare_cosmos_dataset`.
3. Run the smoke path with `verl/experimental/vla/run_pi05_cosmos_sac.sh`.
4. Scale to disaggregated train/env nodes with `verl/experimental/vla/run_pi05_cosmos_sac_disagg.sh`.
