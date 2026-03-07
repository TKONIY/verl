#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG=${IMAGE_TAG:-localhost/verl-multimodality:vllm012-arm64}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DATA_DIR=${DATA_DIR:-$HOME/data/geo3k}
HF_CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}
RUN_ROOT=${RUN_ROOT:-$REPO_DIR/runs/multimodality}
MODE=${MODE:-sync_colocate}
MODEL_SIZE=${MODEL_SIZE:-7b}
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE:-2}
ROLLOUT_N_GPUS_PER_NODE=${ROLLOUT_N_GPUS_PER_NODE:-2}
GLOBAL_PROFILER_TOOL=${GLOBAL_PROFILER_TOOL:-nsys}
RUNTIME_INSTALL=${RUNTIME_INSTALL:-0}

mkdir -p "$HF_CACHE_DIR" "$RUN_ROOT"

INNER_CMD=$(cat <<'INNER'
set -euo pipefail
cd /root/slime
bash scripts/multimodality/install_runtime_stack.sh
export HOME=/root
export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface
export VLLM_USE_V1=1
export WANDB_MODE=offline
export TRAIN_FILES=/root/data/train.parquet
export VAL_FILES=/root/data/test.parquet
export RUN_ROOT=/root/slime/runs/multimodality
bash scripts/multimodality/run_profiled_grpo_vlm.sh
INNER
)

podman-hpc run --rm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  -v "${REPO_DIR}":/root/slime:Z \
  -v "${DATA_DIR}":/root/data:Z \
  -v "${HF_CACHE_DIR}":/root/.cache/huggingface:Z \
  -e MODE="${MODE}" \
  -e MODEL_SIZE="${MODEL_SIZE}" \
  -e NNODES="${NNODES}" \
  -e N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
  -e TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE}" \
  -e ROLLOUT_N_GPUS_PER_NODE="${ROLLOUT_N_GPUS_PER_NODE}" \
  -e GLOBAL_PROFILER_TOOL="${GLOBAL_PROFILER_TOOL}" \
  -e RUNTIME_INSTALL="${RUNTIME_INSTALL}" \
  "${IMAGE_TAG}" \
  bash -lc "${INNER_CMD}"
