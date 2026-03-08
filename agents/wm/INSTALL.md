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

## `cosmos-rl` smoke run in this cluster

For the current GH200 ARM64 environment, the most reproducible minimal `cosmos-rl` path is:

```bash
bash scripts/run_cosmos_rl_smoke_apptainer.sh
```

What the script does:

- uses the existing ARM64 apptainer image from `~/code/verl_docker`
- creates a temporary `--system-site-packages` venv inside the container
- installs only the Python packages that were missing from the base image for a minimal `cosmos-rl` SFT path
- provides wrapper scripts for `redis-server` and `torchrun`
- runs a `1 GPU`, `1 step` SFT smoke example against the local fixture dataset in `third_party/cosmos-rl/tests/data_fixtures/sharegpt52k_small`

## Direct-run verdict for `cosmos-rl`

In the current workspace environment, `cosmos-rl` is **not** directly runnable on the bare host.

Main reasons:

- the host Python environment does not include GPU `torch` and other core runtime packages
- the base ARM64 image is close, but still misses several Python packages required by `cosmos-rl`
- the controller expects a `redis-server` executable
- the `redislite` Redis binary used here rejects the generated `tls-port 0` config entry, so the wrapper strips that line for compatibility

So the practical answer is:

- not direct on bare host
- runnable inside the apptainer image with a small compatibility shim

