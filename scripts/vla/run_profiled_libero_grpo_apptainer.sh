#!/usr/bin/env bash
set -euo pipefail

VERL_DOCKER_ROOT=${VERL_DOCKER_ROOT:-$HOME/code/verl_docker}
SIF_PATH=${SIF_PATH:-$VERL_DOCKER_ROOT/verl_sgl056_arm64_latest.sif}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
HF_CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}
HF_CACHE_MOUNT=/cache/huggingface
DATA_ROOT=${DATA_ROOT:-$HOME/data}
VLA_DATA_DIR=${VLA_DATA_DIR:-$DATA_ROOT/libero_rl}
RUN_ROOT=${RUN_ROOT:-$REPO_DIR/runs/vla}
RUN_NAME=${RUN_NAME:-vla_libero_grpo_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=$RUN_ROOT/$RUN_NAME
HOST_TMPDIR=${HOST_TMPDIR:-$RUN_ROOT/tmp/apptainer}
HOST_CACHEDIR=${HOST_CACHEDIR:-$RUN_ROOT/tmp/apptainer-cache}
BASE_OVERLAY_HOST=${BASE_OVERLAY_HOST:-$VERL_DOCKER_ROOT/runtime_overlays/sglang_cudnn916_py310_arm64_notorch}
BASE_OVERLAY_CONTAINER=/workspace/verl_docker/runtime_overlays/sglang_cudnn916_py310_arm64_notorch
VLA_OVERLAY_HOST=${VLA_OVERLAY_HOST:-$VERL_DOCKER_ROOT/runtime_overlays/vla_py310_arm64_notorch}
VLA_OVERLAY_CONTAINER=/workspace/verl_docker/runtime_overlays/vla_py310_arm64_notorch
LIBERO_SRC_HOST=${LIBERO_SRC_HOST:-$VERL_DOCKER_ROOT/runtime_overlays/libero_src/LIBERO}
LIBERO_SRC_CONTAINER=/workspace/verl_docker/runtime_overlays/libero_src/LIBERO
SHIM_DIR_HOST=${SHIM_DIR_HOST:-$REPO_DIR/scripts/vla}
SHIM_DIR_CONTAINER=/workspace/verl/scripts/vla
CONTAINER_HOME=${CONTAINER_HOME:-/tmp/verl_home}
CONTAINER_CACHE_HOME=${CONTAINER_CACHE_HOME:-$CONTAINER_HOME/.cache}
CONTAINER_TRITON_CACHE=${CONTAINER_TRITON_CACHE:-$CONTAINER_CACHE_HOME/triton}
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
N_ENV_GPUS_PER_NODE=${N_ENV_GPUS_PER_NODE:-2}
ROLLOUT_N=${ROLLOUT_N:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
GLOBAL_PROFILER_TOOL=${GLOBAL_PROFILER_TOOL:-torch}
PROFILE_STEPS=${PROFILE_STEPS:-[1]}
SFT_MODEL_PATH=${SFT_MODEL_PATH:-Haozhan72/Openvla-oft-SFT-libero10-trajall}
LOCALIZE_MODEL_FROM_HF=${LOCALIZE_MODEL_FROM_HF:-1}
DISAGG_SIM_ENABLE=${DISAGG_SIM_ENABLE:-False}
DISAGG_SIM_NNODES=${DISAGG_SIM_NNODES:-1}

mkdir -p "$HF_CACHE_DIR" "$VLA_DATA_DIR" "$RUN_DIR" "$VLA_OVERLAY_HOST" "$HOST_TMPDIR" "$HOST_CACHEDIR"
export TMPDIR="$HOST_TMPDIR"
export APPTAINER_TMPDIR="$HOST_TMPDIR"
export APPTAINER_CACHEDIR="$HOST_CACHEDIR"
export SINGULARITY_TMPDIR="$HOST_TMPDIR"
export SINGULARITY_CACHEDIR="$HOST_CACHEDIR"

INNER=$(cat <<'INNER_EOF'
set -euo pipefail
cd /workspace/verl
export HOME=__CONTAINER_HOME__
export XDG_CACHE_HOME=__CONTAINER_CACHE_HOME__
export FLASHINFER_WORKSPACE_BASE=__CONTAINER_HOME__
export TRITON_CACHE_DIR=__CONTAINER_TRITON_CACHE__
export TMPDIR=${TMPDIR:-__CONTAINER_HOME__/tmp}
export RAY_TMPDIR=${RAY_TMPDIR:-/workspace/verl/raytmp}
export HF_HOME=__HF_CACHE_MOUNT__
export TRANSFORMERS_CACHE=__HF_CACHE_MOUNT__
export WANDB_MODE=offline
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export LIBERO_CONFIG_PATH=$HOME/.libero
export VERL_FILE_LOGGER_PATH=/workspace/verl/runs/vla/__RUN_NAME__/metrics.jsonl
mkdir -p "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: __LIBERO_SRC_CONTAINER__/libero/libero
bddl_files: __LIBERO_SRC_CONTAINER__/libero/libero/bddl_files
init_states: __LIBERO_SRC_CONTAINER__/libero/libero/init_files
datasets: /workspace/data/libero_datasets
assets: __LIBERO_SRC_CONTAINER__/libero/libero/assets
YAML
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$HOME/.cache/flashinfer" "$TRITON_CACHE_DIR" "$TMPDIR" "$RAY_TMPDIR" /workspace/verl/runs/vla/__RUN_NAME__

if ! PYTHONPATH="__LIBERO_SRC_CONTAINER__:__VLA_OVERLAY_CONTAINER__:__BASE_OVERLAY_CONTAINER__" python3 - <<'PY'
import importlib
mods = ["libero.libero", "mujoco", "robosuite", "bddl"]
for mod in mods:
    importlib.import_module(mod)
    print(f"import_ok {mod}")
PY
then
  REPO_DIR=/workspace/verl BASE_OVERLAY=__BASE_OVERLAY_CONTAINER__ LIBERO_SRC_DIR=__LIBERO_SRC_CONTAINER__ bash scripts/vla/setup_vla_runtime_overlay.sh __VLA_OVERLAY_CONTAINER__
fi

export PYTHONPATH=__SHIM_DIR_CONTAINER__:__LIBERO_SRC_CONTAINER__:__VLA_OVERLAY_CONTAINER__:__BASE_OVERLAY_CONTAINER__:/workspace/verl${PYTHONPATH:+:$PYTHONPATH}

if [[ ! -f /workspace/data/libero_rl/train.parquet || ! -f /workspace/data/libero_rl/test.parquet ]]; then
  python3 verl/experimental/vla/prepare_libero_dataset.py --local_save_dir /workspace/data/libero_rl
fi

export SFT_MODEL_PATH=__SFT_MODEL_PATH__
if [[ "__LOCALIZE_MODEL_FROM_HF__" == "1" && "__SFT_MODEL_PATH__" != /* && "__SFT_MODEL_PATH__" != ./* ]]; then
  python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id=os.environ['SFT_MODEL_PATH'], cache_dir=os.environ['HF_HOME'])
with open('/tmp/vla_model_path.txt', 'w') as f:
    f.write(path.strip())
print(path)
PY
  export SFT_MODEL_PATH=$(cat /tmp/vla_model_path.txt)
else
  export SFT_MODEL_PATH=__SFT_MODEL_PATH__
fi

echo "run_name=__RUN_NAME__ sft_model_path=$SFT_MODEL_PATH disagg=__DISAGG_SIM_ENABLE__" | tee /workspace/verl/runs/vla/__RUN_NAME__/train.log
python3 -m verl.experimental.vla.main_ppo \
  data.train_files=/workspace/data/libero_rl/train.parquet \
  data.val_files=/workspace/data/libero_rl/test.parquet \
  data.train_batch_size=__TRAIN_BATCH_SIZE__ \
  data.val_batch_size=__VAL_BATCH_SIZE__ \
  data.max_prompt_length=256 \
  data.max_response_length=128 \
  actor_rollout_ref.rollout.n=__ROLLOUT_N__ \
  env.train.num_envs=__ROLLOUT_N__ \
  env.rollout.pipeline_stage_num=2 \
  env.train.simulator_type=libero \
  env.train.only_eval=False \
  env.train.max_episode_steps=128 \
  env.train.video_cfg.save_video=False \
  env.train.seed=42 \
  env.train.profiler.enable=True \
  +actor_rollout_ref.algorithm=grpo \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.model.path=$SFT_MODEL_PATH \
  actor_rollout_ref.rollout.mode=async_envloop \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.optim.warmup_style=constant \
  actor_rollout_ref.actor.ppo_mini_batch_size=__TRAIN_BATCH_SIZE__ \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.grad_clip=1 \
  actor_rollout_ref.actor.num_images_in_input=1 \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.trust_remote_code=False \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.prompt_length=512 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.adv_estimator=reinforce_plus_plus \
  trainer.logger='["console","file"]' \
  trainer.project_name=vla_profile \
  trainer.experiment_name=__RUN_NAME__ \
  trainer.default_local_dir=/workspace/verl/runs/vla/__RUN_NAME__/checkpoints \
  trainer.n_gpus_per_node=__N_GPUS_PER_NODE__ \
  +trainer.n_env_gpus_per_node=__N_ENV_GPUS_PER_NODE__ \
  +trainer.n_rollout_gpus_per_node=__N_ROLLOUT_GPUS_PER_NODE__ \
  trainer.nnodes=__NNODES__ \
  trainer.save_freq=-1 \
  trainer.test_freq=0 \
  trainer.total_epochs=__TOTAL_EPOCHS__ \
  trainer.val_only=False \
  trainer.total_training_steps=__TOTAL_TRAINING_STEPS__ \
  trainer.val_before_train=False \
  +ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR \
  +ray_kwargs.ray_init.runtime_env.env_vars.HOME=$HOME \
  +ray_kwargs.ray_init.runtime_env.env_vars.XDG_CACHE_HOME=$XDG_CACHE_HOME \
  +ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME=$HF_HOME \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRITON_CACHE_DIR=$TRITON_CACHE_DIR \
  +ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE \
  +ray_kwargs.ray_init.runtime_env.env_vars.TMPDIR=$TMPDIR \
  +ray_kwargs.ray_init.runtime_env.env_vars.RAY_TMPDIR=$RAY_TMPDIR \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$PYTHONPATH \
  env.disagg_sim.enable=__DISAGG_SIM_ENABLE__ \
  env.disagg_sim.nnodes=__DISAGG_SIM_NNODES__ \
  global_profiler.tool=__GLOBAL_PROFILER_TOOL__ \
  global_profiler.steps='__PROFILE_STEPS__' \
  global_profiler.profile_continuous_steps=False \
  global_profiler.save_path=/workspace/verl/runs/vla/__RUN_NAME__/profiles \
  2>&1 | tee -a /workspace/verl/runs/vla/__RUN_NAME__/train.log
INNER_EOF
)

N_ROLLOUT_GPUS_PER_NODE=$((N_GPUS_PER_NODE - N_ENV_GPUS_PER_NODE))
INNER=${INNER//__CONTAINER_HOME__/$CONTAINER_HOME}
INNER=${INNER//__CONTAINER_CACHE_HOME__/$CONTAINER_CACHE_HOME}
INNER=${INNER//__CONTAINER_TRITON_CACHE__/$CONTAINER_TRITON_CACHE}
INNER=${INNER//__HF_CACHE_MOUNT__/$HF_CACHE_MOUNT}
INNER=${INNER//__RUN_NAME__/$RUN_NAME}
INNER=${INNER//__VLA_OVERLAY_CONTAINER__/$VLA_OVERLAY_CONTAINER}
INNER=${INNER//__BASE_OVERLAY_CONTAINER__/$BASE_OVERLAY_CONTAINER}
INNER=${INNER//__LIBERO_SRC_CONTAINER__/$LIBERO_SRC_CONTAINER}
INNER=${INNER//__SHIM_DIR_CONTAINER__/$SHIM_DIR_CONTAINER}
INNER=${INNER//__SFT_MODEL_PATH__/$SFT_MODEL_PATH}
INNER=${INNER//__LOCALIZE_MODEL_FROM_HF__/$LOCALIZE_MODEL_FROM_HF}
INNER=${INNER//__TRAIN_BATCH_SIZE__/$TRAIN_BATCH_SIZE}
INNER=${INNER//__VAL_BATCH_SIZE__/$VAL_BATCH_SIZE}
INNER=${INNER//__ROLLOUT_N__/$ROLLOUT_N}
INNER=${INNER//__N_GPUS_PER_NODE__/$N_GPUS_PER_NODE}
INNER=${INNER//__N_ENV_GPUS_PER_NODE__/$N_ENV_GPUS_PER_NODE}
INNER=${INNER//__N_ROLLOUT_GPUS_PER_NODE__/$N_ROLLOUT_GPUS_PER_NODE}
INNER=${INNER//__NNODES__/$NNODES}
INNER=${INNER//__TOTAL_EPOCHS__/$TOTAL_EPOCHS}
INNER=${INNER//__TOTAL_TRAINING_STEPS__/$TOTAL_TRAINING_STEPS}
INNER=${INNER//__DISAGG_SIM_ENABLE__/$DISAGG_SIM_ENABLE}
INNER=${INNER//__DISAGG_SIM_NNODES__/$DISAGG_SIM_NNODES}
INNER=${INNER//__GLOBAL_PROFILER_TOOL__/$GLOBAL_PROFILER_TOOL}
INNER=${INNER//__PROFILE_STEPS__/$PROFILE_STEPS}

BIND_PATHS="$REPO_DIR:/workspace/verl,$DATA_ROOT:/workspace/data,$HF_CACHE_DIR:$HF_CACHE_MOUNT,$VERL_DOCKER_ROOT:/workspace/verl_docker"
apptainer exec --nv --bind "$BIND_PATHS" "$SIF_PATH" bash -lc "$INNER"
