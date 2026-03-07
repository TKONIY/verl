#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
PARTITION=${PARTITION:-workq}
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-32}
TIME_LIMIT=${TIME_LIMIT:-02:00:00}
MODE=${MODE:-sync_colocate}
MODEL_SIZE=${MODEL_SIZE:-7b}
IMAGE_TAG=${IMAGE_TAG:-localhost/verl-multimodality:vllm012-arm64}

mkdir -p "$REPO_DIR/runs/multimodality/slurm_logs"
JOB_SCRIPT=$(mktemp "$REPO_DIR/runs/multimodality/slurm_logs/${MODE}_${MODEL_SIZE}_XXXXXX.slurm")
cat > "$JOB_SCRIPT" <<SCRIPT
#!/usr/bin/env bash
#SBATCH -J mm_${MODE}_${MODEL_SIZE}
#SBATCH -p ${PARTITION}
#SBATCH -N ${NNODES}
#SBATCH --ntasks=1
#SBATCH --gres=gpu:${N_GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH -t ${TIME_LIMIT}
#SBATCH -o ${REPO_DIR}/runs/multimodality/slurm_logs/%x_%j.out
#SBATCH -e ${REPO_DIR}/runs/multimodality/slurm_logs/%x_%j.err
module load cuda/12.6 nvidia/24.11
cd ${REPO_DIR}
IMAGE_TAG=${IMAGE_TAG} MODE=${MODE} MODEL_SIZE=${MODEL_SIZE} NNODES=${NNODES} N_GPUS_PER_NODE=${N_GPUS_PER_NODE} \
  bash scripts/multimodality/run_profiled_grpo_vlm_podman.sh
SCRIPT
chmod +x "$JOB_SCRIPT"
echo "$JOB_SCRIPT"
sbatch --wait "$JOB_SCRIPT"
