#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
PARTITION=${PARTITION:-workq}
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-32}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
MODE=${MODE:-sync_colocate}
MODEL_SIZE=${MODEL_SIZE:-3b}
IMAGE_TAG=${IMAGE_TAG:-docker.io/mksit/slime_cu129_arm64:latest}
RUNTIME_INSTALL=${RUNTIME_INSTALL:-1}
HF_CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-$HOME/.cache/pip}
DATA_DIR=${DATA_DIR:-$HOME/data/geo3k}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-16}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-2}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
PROFILE_STEPS=${PROFILE_STEPS:-[1]}
GLOBAL_PROFILER_TOOL=${GLOBAL_PROFILER_TOOL:-nsys}
TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE:-2}
ROLLOUT_N_GPUS_PER_NODE=${ROLLOUT_N_GPUS_PER_NODE:-2}

mkdir -p "$REPO_DIR/runs/multimodality/slurm_logs" "$HF_CACHE_DIR" "$PIP_CACHE_DIR"
JOB_SCRIPT=$(mktemp "$REPO_DIR/runs/multimodality/slurm_logs/${MODE}_${MODEL_SIZE}_ray_XXXXXX.slurm")
cat > "$JOB_SCRIPT" <<SCRIPT
#!/usr/bin/env bash
#SBATCH -J mmray_${MODE}_${MODEL_SIZE}
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
visible_devices=$(seq -s, 0 $((${N_GPUS_PER_NODE} - 1)))

echo "Head node: \$head_node"
echo "Head address: \$ip_head"

srun --nodes=1 --ntasks=1 -w "\$head_node" podman-hpc run --rm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --network=host \
  -v /tmp/ray:/tmp/ray:Z \
  -v "${REPO_DIR}":/root/slime:Z \
  -v "${DATA_DIR}":/root/data:Z \
  -v "${HF_CACHE_DIR}":/root/.cache/huggingface:Z \
  -v "${PIP_CACHE_DIR}":/root/.cache/pip:Z \
  -e CUDA_VISIBLE_DEVICES=\$visible_devices -e NVIDIA_VISIBLE_DEVICES=all -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 -e RUNTIME_INSTALL=${RUNTIME_INSTALL} -e RAY_TMPDIR=/tmp/ray ${IMAGE_TAG} bash /root/slime/scripts/multimodality/podman_ray_head.sh "\$head_node_ip" "\$port" "\$SLURM_CPUS_PER_TASK" "${N_GPUS_PER_NODE}" &
sleep 10

worker_num=\$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=\${nodes[\$i]}
  echo "Starting worker on \$node_i"
  srun --nodes=1 --ntasks=1 -w "\$node_i" podman-hpc run --rm \
    --device nvidia.com/gpu=all \
    --ipc=host \
    --network=host \
  -v /tmp/ray:/tmp/ray:Z \
    -v "${REPO_DIR}":/root/slime:Z \
    -v "${DATA_DIR}":/root/data:Z \
    -v "${HF_CACHE_DIR}":/root/.cache/huggingface:Z \
  -v "${PIP_CACHE_DIR}":/root/.cache/pip:Z \
    -e CUDA_VISIBLE_DEVICES=\$visible_devices -e NVIDIA_VISIBLE_DEVICES=all -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 -e RUNTIME_INSTALL=${RUNTIME_INSTALL} -e RAY_TMPDIR=/tmp/ray ${IMAGE_TAG} bash /root/slime/scripts/multimodality/podman_ray_worker.sh "\$ip_head" "\$SLURM_CPUS_PER_TASK" "${N_GPUS_PER_NODE}" &
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
  -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
  -e MODE=${MODE} \
  -e MODEL_SIZE=${MODEL_SIZE} \
  -e NNODES=${NNODES} \
  -e N_GPUS_PER_NODE=${N_GPUS_PER_NODE} \
  -e TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE} \
  -e ROLLOUT_N_GPUS_PER_NODE=${ROLLOUT_N_GPUS_PER_NODE} \
  -e TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES} \
  -e VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES} \
  -e TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
  -e N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT} \
  -e TOTAL_EPOCHS=${TOTAL_EPOCHS} \
  -e PROFILE_STEPS='${PROFILE_STEPS}' \
  -e GLOBAL_PROFILER_TOOL=${GLOBAL_PROFILER_TOOL} \
  -e RUNTIME_INSTALL=${RUNTIME_INSTALL} \
  -e PIP_CACHE_DIR=/root/.cache/pip \
  -e SLURM_CPUS_PER_TASK=\$SLURM_CPUS_PER_TASK \
  -e RAY_ADDRESS=\$ip_head \
  -e N_GPUS_PER_NODE=${N_GPUS_PER_NODE} \
  -e RAY_TMPDIR=/tmp/ray ${IMAGE_TAG} bash /root/slime/scripts/multimodality/podman_train_entry.sh
SCRIPT
chmod +x "$JOB_SCRIPT"
echo "$JOB_SCRIPT"
sbatch --wait "$JOB_SCRIPT"
