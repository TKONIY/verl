#!/usr/bin/env bash
set -euo pipefail
cd /root/slime
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
bash /root/slime/scripts/multimodality/install_runtime_stack.sh
export HOME=/root
export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface
export VLLM_USE_V1=1
export WANDB_MODE=offline
export TRAIN_FILES=/root/data/train.parquet
export VAL_FILES=/root/data/test.parquet
export RUN_ROOT=/root/slime/runs/multimodality
EXTRA_RAY_ARGS=()
if [[ -n "${RAY_ADDRESS:-}" ]]; then
  EXTRA_RAY_ARGS+=("+ray_kwargs.ray_init.address=${RAY_ADDRESS}")
else
  EXTRA_RAY_ARGS+=("ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-32}")
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  EXTRA_RAY_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi
if [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" ]]; then
  EXTRA_RAY_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}")
fi
EXTRA_RAY_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1")
EXTRA_RAY_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0")
EXTRA_RAY_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.RAY_TMPDIR=/tmp/ray")

bash scripts/multimodality/run_profiled_grpo_vlm.sh "${EXTRA_RAY_ARGS[@]}"
