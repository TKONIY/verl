#!/usr/bin/env bash
set -euo pipefail
cd /root/slime

if [[ "${RUNTIME_INSTALL:-1}" == "1" ]]; then
  python -m pip install "nvidia-ml-py>=12.560.30"
  python -m pip install -e '.[vllm,geo]'
  python -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu129 \
    torch==2.9.0+cu129 torchvision==0.24.0 torchaudio==2.9.0
  python -m pip install --force-reinstall \
    numpy==1.26.4 fsspec==2025.10.0 setuptools==80.9.0
else
  python -m pip install --no-deps -e .
fi
