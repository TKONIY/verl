#!/usr/bin/env bash
set -euo pipefail

VERL_DOCKER_ROOT=${VERL_DOCKER_ROOT:-$HOME/code/verl_docker}
SIF_PATH=${SIF_PATH:-$VERL_DOCKER_ROOT/verl_sgl056_arm64_latest.sif}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
DATA_DIR=${DATA_DIR:-$HOME/data/geo3k}
HF_CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}
HF_CACHE_MOUNT=/cache/huggingface
RUN_ROOT=${RUN_ROOT:-$REPO_DIR/runs/multimodality}
MODE=${MODE:-sync_colocate}
MODEL_SIZE=${MODEL_SIZE:-3b}
MODEL_PATH=${MODEL_PATH:-}
ENGINE=${ENGINE:-sglang}
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE:-2}
ROLLOUT_N_GPUS_PER_NODE=${ROLLOUT_N_GPUS_PER_NODE:-2}
GLOBAL_PROFILER_TOOL=${GLOBAL_PROFILER_TOOL:-torch}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-16}}
RAY_ADDRESS=${RAY_ADDRESS:-}
RAY_PRESERVE_VISIBLE_DEVICES=${RAY_PRESERVE_VISIBLE_DEVICES:-0}
RUN_NAME=${RUN_NAME:-${MODE}_${MODEL_SIZE}_${ENGINE}_$(date +%Y%m%d_%H%M%S)}
LOCALIZE_MODEL_FROM_HF=${LOCALIZE_MODEL_FROM_HF:-1}
SGLANG_FIX_CUDNN=${SGLANG_FIX_CUDNN:-1}
RUNTIME_OVERLAY_HOST=${RUNTIME_OVERLAY_HOST:-$VERL_DOCKER_ROOT/runtime_overlays/sglang_cudnn916_py310_arm64_notorch}
RUNTIME_OVERLAY_CONTAINER=/workspace/verl_docker/runtime_overlays/sglang_cudnn916_py310_arm64_notorch
CONTAINER_HOME=${CONTAINER_HOME:-/tmp/verl_home}
CONTAINER_CACHE_HOME=${CONTAINER_CACHE_HOME:-$CONTAINER_HOME/.cache}
CONTAINER_TRITON_CACHE=${CONTAINER_TRITON_CACHE:-$CONTAINER_CACHE_HOME/triton}

mkdir -p "$HF_CACHE_DIR" "$RUN_ROOT/$RUN_NAME" "$RUNTIME_OVERLAY_HOST" "$RUN_ROOT/tmp/pip"

INNER=$(cat <<'INNER_EOF'
set -euo pipefail
cd /workspace/verl
export PYTHONPATH=/workspace/verl${PYTHONPATH:+:$PYTHONPATH}
export HOME=__CONTAINER_HOME__
export XDG_CACHE_HOME=__CONTAINER_CACHE_HOME__
export FLASHINFER_WORKSPACE_BASE=__CONTAINER_HOME__
export TRITON_CACHE_DIR=__CONTAINER_TRITON_CACHE__
export TMPDIR=/workspace/verl/runs/multimodality/tmp/pip
export HF_HOME=__HF_CACHE_MOUNT__
export TRANSFORMERS_CACHE=__HF_CACHE_MOUNT__
export WANDB_MODE=offline
export VLLM_USE_V1=1
export TRAIN_FILES=/workspace/data/train.parquet
export VAL_FILES=/workspace/data/test.parquet
export RUN_ROOT=/workspace/verl/runs/multimodality
export RUN_NAME=__RUN_NAME__
export MODE=__MODE__
export MODEL_SIZE=__MODEL_SIZE__
export MODEL_PATH=__MODEL_PATH__
export ENGINE=__ENGINE__
export NNODES=__NNODES__
export N_GPUS_PER_NODE=__N_GPUS_PER_NODE__
export TRAINER_N_GPUS_PER_NODE=__TRAINER_N_GPUS_PER_NODE__
export ROLLOUT_N_GPUS_PER_NODE=__ROLLOUT_N_GPUS_PER_NODE__
export GLOBAL_PROFILER_TOOL=__GLOBAL_PROFILER_TOOL__
export LOCALIZE_MODEL_FROM_HF=__LOCALIZE_MODEL_FROM_HF__
export SGLANG_FIX_CUDNN=__SGLANG_FIX_CUDNN__
export RUNTIME_OVERLAY=__RUNTIME_OVERLAY_CONTAINER__
export RAY_NUM_CPUS=__RAY_NUM_CPUS__
export RAY_ADDRESS=__RAY_ADDRESS__
export RAY_PRESERVE_VISIBLE_DEVICES=__RAY_PRESERVE_VISIBLE_DEVICES__
if [[ "$RAY_PRESERVE_VISIBLE_DEVICES" == "1" ]]; then
  export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
  export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
fi
export RAY_TMPDIR=/workspace/verl/raytmp
mkdir -p "$RAY_TMPDIR" /workspace/verl/runs/multimodality/__RUN_NAME__ __HF_CACHE_MOUNT__ "$TMPDIR" "$RUNTIME_OVERLAY" "$HOME" "$XDG_CACHE_HOME" "$HOME/.cache/flashinfer" "$TRITON_CACHE_DIR"
LOG=/workspace/verl/runs/multimodality/__RUN_NAME__/train.log

if [[ "$ENGINE" == "sglang" && "$SGLANG_FIX_CUDNN" == "1" ]]; then
  if ! python3 - <<'PY'
import os
import sys
import importlib.metadata as metadata
sys.path.insert(0, os.environ['RUNTIME_OVERLAY'])
try:
    version = metadata.version('nvidia-cudnn-cu12')
    major, minor, *_ = [int(part) for part in version.split('.')]
    print(f'overlay_cudnn_version={version}')
    raise SystemExit(0 if (major, minor) >= (9, 15) else 1)
except Exception as exc:
    print(f'overlay_cudnn_missing={exc}')
    raise SystemExit(1)
PY
  then
    echo "Installing CuDNN overlay into $RUNTIME_OVERLAY" | tee -a "$LOG"
    python3 -m pip install --target "$RUNTIME_OVERLAY" --upgrade --no-cache-dir nvidia-cudnn-cu12==9.16.0.29 | tee -a "$LOG"
  fi
fi

if [[ "$MODE" == "fully_async_disaggregate" ]]; then
  if ! python3 - <<'PY'
import os
import sys
sys.path.insert(0, os.environ['RUNTIME_OVERLAY'])
try:
    import cupy  # noqa: F401
    print('overlay_cupy=present')
    raise SystemExit(0)
except Exception as exc:
    print(f'overlay_cupy_missing={exc}')
    raise SystemExit(1)
PY
  then
    echo "Installing CuPy overlay into $RUNTIME_OVERLAY" | tee -a "$LOG"
    python3 -m pip install --target "$RUNTIME_OVERLAY" --upgrade --no-cache-dir cupy-cuda12x | tee -a "$LOG"
  fi
fi

export PYTHONPATH="$RUNTIME_OVERLAY:$PYTHONPATH"

python3 - <<'PY' | tee -a "$LOG"
import os
import sys
import importlib.metadata as metadata
import torch
sys.path.insert(0, os.environ['RUNTIME_OVERLAY'])
try:
    print(f'active_overlay_cudnn={metadata.version("nvidia-cudnn-cu12")}')
except Exception as exc:
    print(f'active_overlay_cudnn=missing:{exc}')
try:
    import cupy  # noqa: F401
    print('active_overlay_cupy=present')
except Exception as exc:
    print(f'active_overlay_cupy=missing:{exc}')
print(f'overlay_lib_dir={os.path.join(os.environ["RUNTIME_OVERLAY"], "nvidia", "cudnn", "lib")}')
print(f'torch_cudnn_version={torch.backends.cudnn.version()}')
PY

if [[ -z "$MODEL_PATH" ]]; then
  case "$MODEL_SIZE" in
    3b) MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct ;;
    7b) MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct ;;
    32b) MODEL_PATH=Qwen/Qwen2.5-VL-32B-Instruct ;;
  esac
fi
if [[ "$LOCALIZE_MODEL_FROM_HF" == "1" && "$MODEL_PATH" != /* && "$MODEL_PATH" != ./* ]]; then
  python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
repo_id = os.environ['MODEL_PATH']
path = snapshot_download(repo_id=repo_id, cache_dir=os.environ['HF_HOME'])
print(path)
with open('/tmp/multimodal_model_path.txt', 'w') as f:
    f.write(path.strip())
PY
  export MODEL_PATH=$(cat /tmp/multimodal_model_path.txt)
  echo "localized_model_path=$MODEL_PATH" | tee -a "$LOG"
fi

echo "run_name=$RUN_NAME mode=$MODE model_size=$MODEL_SIZE engine=$ENGINE model_path=$MODEL_PATH" | tee -a "$LOG"
EXTRA_RAY_ARGS=(
  "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=\"$PYTHONPATH\""
  "+ray_kwargs.ray_init.runtime_env.env_vars.HOME=\"$HOME\""
  "+ray_kwargs.ray_init.runtime_env.env_vars.XDG_CACHE_HOME=\"$XDG_CACHE_HOME\""
  "+ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_WORKSPACE_BASE=\"$HOME\""
  "+ray_kwargs.ray_init.runtime_env.env_vars.TRITON_CACHE_DIR=\"$TRITON_CACHE_DIR\""
  "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME=\"$HF_HOME\""
  "+ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_CACHE=\"$TRANSFORMERS_CACHE\""
  "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE=\"$WANDB_MODE\""
  "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1=\"$VLLM_USE_V1\""
  "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_TMPDIR=\"$RAY_TMPDIR\""
)
if [[ "$RAY_PRESERVE_VISIBLE_DEVICES" == "1" ]]; then
  EXTRA_RAY_ARGS+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=\"1\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=\"0\""
  )
fi
if [[ -n "$RAY_ADDRESS" ]]; then
  EXTRA_RAY_ARGS+=(+ray_kwargs.ray_init.address=$RAY_ADDRESS)
else
  EXTRA_RAY_ARGS+=(+ray_kwargs.ray_init.num_cpus=$RAY_NUM_CPUS)
fi
bash scripts/multimodality/run_profiled_grpo_vlm.sh "${EXTRA_RAY_ARGS[@]}" 2>&1 | tee -a "$LOG"
test ${PIPESTATUS[0]} -eq 0
INNER_EOF
)

INNER=${INNER//__HF_CACHE_MOUNT__/$HF_CACHE_MOUNT}
INNER=${INNER//__RUN_NAME__/$RUN_NAME}
INNER=${INNER//__MODE__/$MODE}
INNER=${INNER//__MODEL_SIZE__/$MODEL_SIZE}
INNER=${INNER//__MODEL_PATH__/$MODEL_PATH}
INNER=${INNER//__ENGINE__/$ENGINE}
INNER=${INNER//__NNODES__/$NNODES}
INNER=${INNER//__N_GPUS_PER_NODE__/$N_GPUS_PER_NODE}
INNER=${INNER//__TRAINER_N_GPUS_PER_NODE__/$TRAINER_N_GPUS_PER_NODE}
INNER=${INNER//__ROLLOUT_N_GPUS_PER_NODE__/$ROLLOUT_N_GPUS_PER_NODE}
INNER=${INNER//__GLOBAL_PROFILER_TOOL__/$GLOBAL_PROFILER_TOOL}
INNER=${INNER//__LOCALIZE_MODEL_FROM_HF__/$LOCALIZE_MODEL_FROM_HF}
INNER=${INNER//__SGLANG_FIX_CUDNN__/$SGLANG_FIX_CUDNN}
INNER=${INNER//__RUNTIME_OVERLAY_CONTAINER__/$RUNTIME_OVERLAY_CONTAINER}
INNER=${INNER//__RAY_NUM_CPUS__/$RAY_NUM_CPUS}
INNER=${INNER//__RAY_ADDRESS__/$RAY_ADDRESS}
INNER=${INNER//__RAY_PRESERVE_VISIBLE_DEVICES__/$RAY_PRESERVE_VISIBLE_DEVICES}
INNER=${INNER//__CONTAINER_HOME__/$CONTAINER_HOME}
INNER=${INNER//__CONTAINER_CACHE_HOME__/$CONTAINER_CACHE_HOME}
INNER=${INNER//__CONTAINER_TRITON_CACHE__/$CONTAINER_TRITON_CACHE}

BIND_PATHS="$REPO_DIR:/workspace/verl,$DATA_DIR:/workspace/data,$HF_CACHE_DIR:$HF_CACHE_MOUNT,$VERL_DOCKER_ROOT:/workspace/verl_docker"
if [[ "$ENGINE" == "sglang" && "$SGLANG_FIX_CUDNN" == "1" ]]; then
  BIND_PATHS+=",$RUNTIME_OVERLAY_HOST/nvidia/cudnn:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn"
  BIND_PATHS+=",$RUNTIME_OVERLAY_HOST/nvidia/cublas:/usr/local/lib/python3.12/dist-packages/nvidia/cublas"
fi

apptainer exec --nv --writable-tmpfs \
  --bind "$BIND_PATHS" \
  "$SIF_PATH" \
  bash -lc "$INNER"
