#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
PARTITION=${PARTITION:-workq}
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-32}
MEM=${MEM:-120G}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
RUN_NAME=${RUN_NAME:-vla_libero_grpo_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$REPO_DIR/runs/vla/slurm_logs"
JOB_SCRIPT=$(mktemp "$REPO_DIR/runs/vla/slurm_logs/vla_libero_grpo_XXXXXX.slurm")
cat > "$JOB_SCRIPT" <<EOF2
#!/usr/bin/env bash
#SBATCH -J vla_libero_grpo
#SBATCH -p ${PARTITION}
#SBATCH -N ${NNODES}
#SBATCH --ntasks=1
#SBATCH --gres=gpu:${N_GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH -t ${TIME_LIMIT}
#SBATCH -o ${REPO_DIR}/runs/vla/slurm_logs/%x_%j.out
#SBATCH -e ${REPO_DIR}/runs/vla/slurm_logs/%x_%j.err
set -euo pipefail
module load cuda/12.6 nvidia/24.11
cd ${REPO_DIR}
export RUN_NAME=${RUN_NAME}
PYTHONUNBUFFERED=1 bash scripts/vla/run_profiled_libero_grpo_apptainer.sh
EOF2
chmod +x "$JOB_SCRIPT"
echo "$JOB_SCRIPT"
sbatch --wait "$JOB_SCRIPT"
