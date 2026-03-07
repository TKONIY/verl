#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
PARTITION=${PARTITION:-workq}
NNODES=${NNODES:-8}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-32}
MEM=${MEM:-220G}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
MODE=${MODE:-sync_colocate}
MODEL_SIZE=${MODEL_SIZE:-3b}
ENGINE=${ENGINE:-sglang}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-32}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-1}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
PROFILE_STEPS=${PROFILE_STEPS:-[1]}
GLOBAL_PROFILER_TOOL=${GLOBAL_PROFILER_TOOL:-torch}
LOCALIZE_MODEL_FROM_HF=${LOCALIZE_MODEL_FROM_HF:-1}
TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE:-2}
ROLLOUT_N_GPUS_PER_NODE=${ROLLOUT_N_GPUS_PER_NODE:-2}
VERL_DOCKER_ROOT=${VERL_DOCKER_ROOT:-$HOME/code/verl_docker}
SIF_PATH=${SIF_PATH:-$VERL_DOCKER_ROOT/verl_sgl056_arm64_latest.sif}
DATA_DIR=${DATA_DIR:-$HOME/data/geo3k}
HF_CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}
RUN_NAME=${RUN_NAME:-${MODE}_${MODEL_SIZE}_${ENGINE}_${NNODES}n_$(date +%Y%m%d_%H%M%S)}
CONTAINER_HOME=${CONTAINER_HOME:-/tmp/verl_home}
CONTAINER_CACHE_HOME=${CONTAINER_CACHE_HOME:-$CONTAINER_HOME/.cache}
CONTAINER_TRITON_CACHE=${CONTAINER_TRITON_CACHE:-$CONTAINER_CACHE_HOME/triton}
RAY_PRESERVE_VISIBLE_DEVICES=${RAY_PRESERVE_VISIBLE_DEVICES:-0}

mkdir -p "$REPO_DIR/runs/multimodality/slurm_logs" "$HF_CACHE_DIR"
JOB_SCRIPT=$(mktemp "$REPO_DIR/runs/multimodality/slurm_logs/${MODE}_${MODEL_SIZE}_${ENGINE}_rayappt_XXXXXX.slurm")
cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH -J mmrayappt_${MODE}_${MODEL_SIZE}_${ENGINE}
#SBATCH -p ${PARTITION}
#SBATCH -N ${NNODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${N_GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH -t ${TIME_LIMIT}
#SBATCH -o ${REPO_DIR}/runs/multimodality/slurm_logs/%x_%j.out
#SBATCH -e ${REPO_DIR}/runs/multimodality/slurm_logs/%x_%j.err
set -euo pipefail
module load cuda/12.6 nvidia/24.11
cd ${REPO_DIR}
mkdir -p /tmp/ray ${CONTAINER_HOME} ${CONTAINER_CACHE_HOME} ${CONTAINER_HOME}/.cache/flashinfer ${CONTAINER_TRITON_CACHE}
export RAY_TMPDIR=/tmp/ray

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
echo "Head node: \$head_node"
echo "Head address: \$ip_head"

BIND_PATHS="${REPO_DIR}:/workspace/verl,${DATA_DIR}:/workspace/data,${HF_CACHE_DIR}:/cache/huggingface,${VERL_DOCKER_ROOT}:/workspace/verl_docker"
OVL="${VERL_DOCKER_ROOT}/runtime_overlays/sglang_cudnn916_py310_arm64_notorch"
if [[ -d "\$OVL/nvidia/cudnn" ]]; then
  BIND_PATHS+=",\$OVL/nvidia/cudnn:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn"
fi
if [[ -d "\$OVL/nvidia/cublas" ]]; then
  BIND_PATHS+=",\$OVL/nvidia/cublas:/usr/local/lib/python3.12/dist-packages/nvidia/cublas"
fi

srun --nodes=1 --ntasks=1 -w "\$head_node" /usr/bin/env RAY_TMPDIR=/tmp/ray apptainer exec --nv --bind "\$BIND_PATHS" "${SIF_PATH}" bash -lc 'export HOME=${CONTAINER_HOME}; export XDG_CACHE_HOME=${CONTAINER_CACHE_HOME}; export FLASHINFER_WORKSPACE_BASE=${CONTAINER_HOME}; export TRITON_CACHE_DIR=${CONTAINER_TRITON_CACHE}; export RAY_TMPDIR=/tmp/ray; if [[ "${RAY_PRESERVE_VISIBLE_DEVICES}" == "1" ]]; then export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1; export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0; fi; mkdir -p "\$HOME" "\$XDG_CACHE_HOME" "\$HOME/.cache/flashinfer" "\$TRITON_CACHE_DIR" /tmp/ray && ray start --head --node-ip-address="'"\$head_node_ip"'" --port='"\$port"' --temp-dir /tmp/ray --num-cpus "\$SLURM_CPUS_PER_TASK" --num-gpus "${N_GPUS_PER_NODE}" --block' &
sleep 10

worker_num=\$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=\${nodes[\$i]}
  echo "Starting worker on \$node_i"
  srun --nodes=1 --ntasks=1 -w "\$node_i" /usr/bin/env RAY_TMPDIR=/tmp/ray apptainer exec --nv --bind "\$BIND_PATHS" "${SIF_PATH}" bash -lc 'export HOME=${CONTAINER_HOME}; export XDG_CACHE_HOME=${CONTAINER_CACHE_HOME}; export FLASHINFER_WORKSPACE_BASE=${CONTAINER_HOME}; export TRITON_CACHE_DIR=${CONTAINER_TRITON_CACHE}; export RAY_TMPDIR=/tmp/ray; if [[ "${RAY_PRESERVE_VISIBLE_DEVICES}" == "1" ]]; then export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1; export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0; fi; mkdir -p "\$HOME" "\$XDG_CACHE_HOME" "\$HOME/.cache/flashinfer" "\$TRITON_CACHE_DIR" /tmp/ray && ray start --address "'"\$ip_head"'" --temp-dir /tmp/ray --num-cpus "\$SLURM_CPUS_PER_TASK" --num-gpus "${N_GPUS_PER_NODE}" --block' &
  sleep 5
done

export MODE=${MODE}
export MODEL_SIZE=${MODEL_SIZE}
export ENGINE=${ENGINE}
export NNODES=${NNODES}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE}
export TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE}
export ROLLOUT_N_GPUS_PER_NODE=${ROLLOUT_N_GPUS_PER_NODE}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
export GEN_BATCH_SIZE=${GEN_BATCH_SIZE}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT}
export TOTAL_EPOCHS=${TOTAL_EPOCHS}
export PROFILE_STEPS='${PROFILE_STEPS}'
export GLOBAL_PROFILER_TOOL=${GLOBAL_PROFILER_TOOL}
export LOCALIZE_MODEL_FROM_HF=${LOCALIZE_MODEL_FROM_HF}
export RUN_NAME=${RUN_NAME}
export VERL_DOCKER_ROOT=${VERL_DOCKER_ROOT}
export SIF_PATH=${SIF_PATH}
export DATA_DIR=${DATA_DIR}
export HF_CACHE_DIR=${HF_CACHE_DIR}
export RAY_ADDRESS=\$ip_head
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
export HOME=${CONTAINER_HOME}
export XDG_CACHE_HOME=${CONTAINER_CACHE_HOME}
export FLASHINFER_WORKSPACE_BASE=${CONTAINER_HOME}
export TRITON_CACHE_DIR=${CONTAINER_TRITON_CACHE}
export RAY_PRESERVE_VISIBLE_DEVICES=${RAY_PRESERVE_VISIBLE_DEVICES}

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "\$head_node" bash scripts/multimodality/run_profiled_grpo_vlm_apptainer.sh
EOF
chmod +x "$JOB_SCRIPT"
echo "$JOB_SCRIPT"
sbatch --wait "$JOB_SCRIPT"
