#!/usr/bin/env bash
set -euo pipefail
cd /root/slime
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
bash /root/slime/scripts/multimodality/install_runtime_stack.sh
ray stop --force || true
ray start --head --node-ip-address="$1" --port="$2" --dashboard-host=0.0.0.0 --num-cpus "$3" --num-gpus "$4" --block
