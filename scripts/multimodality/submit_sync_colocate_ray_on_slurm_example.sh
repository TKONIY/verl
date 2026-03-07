#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
PARTITION=${PARTITION:-workq}
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-32}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
IMAGE_TAG=${IMAGE_TAG:-docker.io/mksit/slime_cu129_arm64:latest}
RUNTIME_INSTALL=${RUNTIME_INSTALL:-1}
HF_CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-$HOME/.cache/pip}
DATA_DIR=${DATA_DIR:-$HOME/data/geo3k}
MODEL_SIZE=${MODEL_SIZE:-3b}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-16}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-2}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
PROFILE_STEPS=${PROFILE_STEPS:-[1]}
GLOBAL_PROFILER_TOOL=${GLOBAL_PROFILER_TOOL:-nsys}
RUN_NAME=${RUN_NAME:-sync_colocate_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)}

case "${MODEL_SIZE}" in
  3b)
    MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
    ROLLOUT_TP=${ROLLOUT_TP:-1}
    ACTOR_MICRO_BATCH=${ACTOR_MICRO_BATCH:-4}
    LOGPROB_MICRO_BATCH=${LOGPROB_MICRO_BATCH:-4}
    GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.65}
    ;;
  7b)
    MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}
    ROLLOUT_TP=${ROLLOUT_TP:-2}
    ACTOR_MICRO_BATCH=${ACTOR_MICRO_BATCH:-2}
    LOGPROB_MICRO_BATCH=${LOGPROB_MICRO_BATCH:-2}
    GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}
    ;;
  32b)
    MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-32B-Instruct}
    ROLLOUT_TP=${ROLLOUT_TP:-4}
    ACTOR_MICRO_BATCH=${ACTOR_MICRO_BATCH:-1}
    LOGPROB_MICRO_BATCH=${LOGPROB_MICRO_BATCH:-1}
    GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.45}
    ;;
  *)
    echo "Unsupported MODEL_SIZE=${MODEL_SIZE}" >&2
    exit 1
    ;;
esac

mkdir -p "$REPO_DIR/runs/multimodality/slurm_logs" "$HF_CACHE_DIR" "$PIP_CACHE_DIR"
JOB_SCRIPT=$(mktemp "$REPO_DIR/runs/multimodality/slurm_logs/sync_colocate_${MODEL_SIZE}_example_XXXXXX.slurm")
cat > "$JOB_SCRIPT" <<SCRIPT
#!/usr/bin/env bash
#SBATCH -J mmray_sync_colocate_${MODEL_SIZE}
#SBATCH -p ${PARTITION}
#SBATCH -N ${NNODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${N_GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH -t ${TIME_LIMIT}
#SBATCH -o ${REPO_DIR}/runs/multimodality/slurm_logs/%x_%j.out
#SBATCH -e ${REPO_DIR}/runs/multimodality/slurm_logs/%x_%j.err

module load cuda/12.6 nvidia/24.11
set -euo pipefail
cd ${REPO_DIR}
export RUNTIME_INSTALL=${RUNTIME_INSTALL}
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
mkdir -p /tmp/ray

nodes=(\$(scontrol show hostnames "\$SLURM_JOB_NODELIST"))
head_node=\${nodes[0]}
head_node_ip=\$(srun --nodes=1 --ntasks=1 -w "\$head_node" hostname --ip-address)
if [[ "\$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"\$head_node_ip"
  if [[ \${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=\${ADDR[1]}
  else
    head_node_ip=\${ADDR[0]}
  fi
fi
port=6379
ip_head="\$head_node_ip:\$port"
export ip_head
visible_devices=$(seq -s, 0 $((${N_GPUS_PER_NODE} - 1)))

echo "Head node: \$head_node"
echo "IP Head: \$ip_head"
printenv | sort

srun --nodes=1 --ntasks=1 -w "\$head_node" podman-hpc run --rm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --network=host \
  -v /tmp/ray:/tmp/ray:Z \
  -v "${REPO_DIR}":/root/slime:Z \
  -v "${DATA_DIR}":/root/data:Z \
  -v "${HF_CACHE_DIR}":/root/.cache/huggingface:Z \
  -v "${PIP_CACHE_DIR}":/root/.cache/pip:Z \
  -e CUDA_VISIBLE_DEVICES=\$visible_devices \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
  -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  -e RAY_TMPDIR=/tmp/ray \
  -e RUNTIME_INSTALL=${RUNTIME_INSTALL} \
  ${IMAGE_TAG} bash /root/slime/scripts/multimodality/podman_ray_head.sh "\$head_node_ip" "\$port" "\$SLURM_CPUS_PER_TASK" "${N_GPUS_PER_NODE}" &
sleep 10

worker_num=\$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=\${nodes[\$i]}
  echo "Starting WORKER \$i at \$node_i"
  srun --nodes=1 --ntasks=1 -w "\$node_i" podman-hpc run --rm \
    --device nvidia.com/gpu=all \
    --ipc=host \
    --network=host \
    -v /tmp/ray:/tmp/ray:Z \
    -v "${REPO_DIR}":/root/slime:Z \
    -v "${DATA_DIR}":/root/data:Z \
    -v "${HF_CACHE_DIR}":/root/.cache/huggingface:Z \
    -v "${PIP_CACHE_DIR}":/root/.cache/pip:Z \
    -e CUDA_VISIBLE_DEVICES=\$visible_devices \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
    -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
    -e RAY_TMPDIR=/tmp/ray \
    -e RUNTIME_INSTALL=${RUNTIME_INSTALL} \
    ${IMAGE_TAG} bash /root/slime/scripts/multimodality/podman_ray_worker.sh "\$ip_head" "\$SLURM_CPUS_PER_TASK" "${N_GPUS_PER_NODE}" &
  sleep 5
done

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "\$head_node" podman-hpc run --rm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --network=host \
  -v /tmp/ray:/tmp/ray:Z \
  -v "${REPO_DIR}":/root/slime:Z \
  -v "${DATA_DIR}":/root/data:Z \
  -v "${HF_CACHE_DIR}":/root/.cache/huggingface:Z \
  -v "${PIP_CACHE_DIR}":/root/.cache/pip:Z \
  -e CUDA_VISIBLE_DEVICES=\$visible_devices \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
  -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  -e RAY_TMPDIR=/tmp/ray \
  -e RUNTIME_INSTALL=${RUNTIME_INSTALL} \
  -e RAY_ADDRESS=\$ip_head \
  -e HOME=/root \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
  -e VLLM_USE_V1=1 \
  -e WANDB_MODE=offline \
  ${IMAGE_TAG} bash -lc 'cd /root/slime && \
    mkdir -p /root/slime/runs/multimodality/${RUN_NAME} && \\
    bash /root/slime/scripts/multimodality/install_runtime_stack.sh && \
    python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files=/root/data/train.parquet \
      data.val_files=/root/data/test.parquet \
      data.image_key=images \
      data.max_prompt_length=1024 \
      data.max_response_length=2048 \
      data.filter_overlong_prompts=True \
      data.truncation=error \
      data.train_max_samples=${TRAIN_MAX_SAMPLES} \
      data.val_max_samples=${VAL_MAX_SAMPLES} \
      data.train_batch_size=${TRAIN_BATCH_SIZE} \
      actor_rollout_ref.model.path=${MODEL_PATH} \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.kl_loss_coef=0.01 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.actor.entropy_coeff=0 \
      actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BATCH} \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH} \
      actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
      actor_rollout_ref.rollout.name=vllm \
      +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
      actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
      actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH} \
      algorithm.use_kl_in_reward=False \
      trainer.critic_warmup=0 \
      trainer.logger='"'"'["console","file"]'"'"' \
      trainer.project_name=verl_multimodal_profile \
      trainer.experiment_name=${RUN_NAME} \
      trainer.total_epochs=${TOTAL_EPOCHS} \
      trainer.save_freq=-1 \
      trainer.test_freq=0 \
      trainer.val_before_train=False \
      trainer.nnodes=${NNODES} \
      trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
      global_profiler.tool=${GLOBAL_PROFILER_TOOL} \
      global_profiler.steps='"'"'${PROFILE_STEPS}'"'"' \
      global_profiler.profile_continuous_steps=False \
      global_profiler.save_path=/root/slime/runs/multimodality/${RUN_NAME}/profiles \
      +ray_kwargs.ray_init.address=\$RAY_ADDRESS \
      +ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=\"1\" \
      +ray_kwargs.ray_init.runtime_env.env_vars.RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=\"0\" \
      +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_VISIBLE_DEVICES=\"\$CUDA_VISIBLE_DEVICES\" \
      trainer.default_local_dir=checkpoints/verl_multimodal_profile/${RUN_NAME} \
      2>&1 | tee /root/slime/runs/multimodality/${RUN_NAME}/train.log'
SCRIPT
chmod +x "$JOB_SCRIPT"
echo "$JOB_SCRIPT"
sbatch --wait "$JOB_SCRIPT"
