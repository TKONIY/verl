#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE_PATH=${VERL_APPTAINER_IMAGE:-/home/u5gl/yangshen.u5gl/code/verl_docker/verl_sgl056_arm64_latest.sif}
VENV_DIR=${COSMOS_RL_VENV_DIR:-/tmp/cosmosrl_run_venv}
WRAP_DIR=${COSMOS_RL_WRAP_DIR:-/tmp/cosmos_bin}
LOG_DIR=${COSMOS_RL_LOG_DIR:-/tmp/cosmos_rl_smoke_logs}
OUT_DIR=${COSMOS_RL_OUT_DIR:-/tmp/cosmos_rl_smoke_output}
HF_CACHE_DIR=${HF_HOME:-/tmp/hf_home_cosmosrl}
MODEL_NAME=${COSMOS_RL_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
MAX_STEPS=${COSMOS_RL_MAX_STEPS:-1}
CONFIG_PATH=${COSMOS_RL_CONFIG_PATH:-/tmp/cosmos_rl_qwen2p5_0p5b_smoke.toml}

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Missing apptainer image: ${IMAGE_PATH}" >&2
  exit 1
fi

if [[ ! -d "${ROOT_DIR}/third_party/cosmos-rl" ]]; then
  echo "Missing third_party/cosmos-rl; run git submodule update --init --recursive third_party/cosmos-rl" >&2
  exit 1
fi

cat > "${CONFIG_PATH}" <<EOF
redis = "12800"

[train]
resume = false
epoch = 1
max_num_steps = ${MAX_STEPS}
output_dir = "${OUT_DIR}"
epsilon = 1e-6
optm_name = "AdamW"
optm_lr = 1e-6
optm_impl = "for-loop"
optm_weight_decay = 0.01
optm_betas = [0.9, 0.999]
optm_warmup_steps = 1
optm_grad_norm_clip = 1.0
async_tp_enabled = false
compile = false
param_dtype = "bfloat16"
fsdp_reduce_dtype = "float32"
fsdp_offload = false
fsdp_reshard_after_forward = "default"
train_batch_per_replica = 1
sync_weight_interval = 1

[policy]
model_name_or_path = "${MODEL_NAME}"
model_max_length = 256
model_gradient_checkpointing = false

[logging]
logger = ["console"]
project_name = "cosmos_rl"
experiment_name = "cosmos_rl_qwen2p5_smoke"

[train.train_policy]
type = "sft"
dataset.name = "/workspace/verl/third_party/cosmos-rl/tests/data_fixtures/sharegpt52k_small"
dataset.subset = ""
dataset.split = "train"
conversation_column_name = "conversation"
mini_batch = 1
dataloader_shuffle = false
dataloader_drop_last = false

[validation]
enable = false

[train.ckpt]
enable_checkpoint = false
save_freq = 100
save_mode = "async"

[policy.parallelism]
n_init_replicas = 1
tp_size = 1
cp_size = 1
dp_shard_size = 1
pp_size = 1
dp_replicate_size = 1
EOF

apptainer exec --nv -B "${ROOT_DIR}:/workspace/verl" "${IMAGE_PATH}" bash -lc "
set -euo pipefail
python -m venv --system-site-packages '${VENV_DIR}'
source '${VENV_DIR}/bin/activate'
if ! python - <<'PY'
for name in ['toml', 'strenum', 'redis', 'boto3', 'diffusers', 'imageio', 'redislite']:
    __import__(name)
PY
then
  python -m pip install --upgrade pip
  python -m pip install toml strenum redis boto3 diffusers imageio redislite
fi
mkdir -p '${WRAP_DIR}'
cat > '${WRAP_DIR}/redis-server' <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
REAL='${VENV_DIR}/lib/python3.12/site-packages/redislite/bin/redis-server'
if [ \"\$#\" -ge 1 ] && [ -f \"\$1\" ]; then
  cfg=\"\$1\"
  patched=\$(mktemp /tmp/cosmos_redis_XXXX.conf)
  grep -v '^tls-port ' \"\$cfg\" > \"\$patched\"
  shift
  exec \"\$REAL\" \"\$patched\" \"\$@\"
else
  exec \"\$REAL\" \"\$@\"
fi
EOF
chmod +x '${WRAP_DIR}/redis-server'
ln -sf '${VENV_DIR}/bin/redis-cli' '${WRAP_DIR}/redis-cli'
cat > '${WRAP_DIR}/torchrun' <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exec python -m torch.distributed.run \"\$@\"
EOF
chmod +x '${WRAP_DIR}/torchrun'
cd /workspace/verl
export PATH='${WRAP_DIR}':\$PATH
export PYTHONPATH=/workspace/verl/third_party/cosmos-rl:\${PYTHONPATH:-}
export HF_HOME='${HF_CACHE_DIR}'
export TOKENIZERS_PARALLELISM=false
python -m cosmos_rl.launcher.launch_all \
  --config '${CONFIG_PATH}' \
  --log-dir '${LOG_DIR}' \
  cosmos_rl.tools.dataset.dummy_sft
"
