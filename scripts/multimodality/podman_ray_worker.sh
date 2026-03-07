#!/usr/bin/env bash
set -euo pipefail
cd /root/slime
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
bash /root/slime/scripts/multimodality/install_runtime_stack.sh
ray stop --force || true
ray start --address "$1" --num-cpus "$2" --num-gpus "$3" --block
