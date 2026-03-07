#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
PARTITION=${PARTITION:-workq}
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
TIME_LIMIT=${TIME_LIMIT:-00:30:00}
IMAGE_TAG=${IMAGE_TAG:-docker.io/mksit/slime_cu129_arm64:latest}
RUNTIME_INSTALL=${RUNTIME_INSTALL:-1}
HF_CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-$HOME/.cache/pip}
mkdir -p "$REPO_DIR/runs/multimodality/slurm_logs" "$HF_CACHE_DIR" "$PIP_CACHE_DIR"
JOB_SCRIPT=$(mktemp "$REPO_DIR/runs/multimodality/slurm_logs/ray_probe_XXXXXX.slurm")
cat > "$JOB_SCRIPT" <<SCRIPT
#!/usr/bin/env bash
#SBATCH -J mmray_probe
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
visible_devices=$(seq -s, 0 $((${N_GPUS_PER_NODE} - 1)))

echo "Head node: \$head_node"
echo "IP Head: \$ip_head"

srun --nodes=1 --ntasks=1 -w "\$head_node" podman-hpc run --rm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --network=host \
  -v /tmp/ray:/tmp/ray:Z \
  -v "${REPO_DIR}":/root/slime:Z \
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

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "\$head_node" podman-hpc run --rm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --network=host \
  -v /tmp/ray:/tmp/ray:Z \
  -v "${REPO_DIR}":/root/slime:Z \
  -v "${HF_CACHE_DIR}":/root/.cache/huggingface:Z \
  -v "${PIP_CACHE_DIR}":/root/.cache/pip:Z \
  -e CUDA_VISIBLE_DEVICES=\$visible_devices \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
  -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  -e RAY_TMPDIR=/tmp/ray \
  -e RAY_ADDRESS=\$ip_head \
  -e RUNTIME_INSTALL=${RUNTIME_INSTALL} \
  ${IMAGE_TAG} bash -lc 'cd /root/slime && bash /root/slime/scripts/multimodality/install_runtime_stack.sh && python scripts/multimodality/ray_actor_env_probe.py'
SCRIPT
chmod +x "$JOB_SCRIPT"
echo "$JOB_SCRIPT"
sbatch --wait "$JOB_SCRIPT"
