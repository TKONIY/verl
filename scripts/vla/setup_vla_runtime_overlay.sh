#!/usr/bin/env bash
set -euo pipefail

OVERLAY_DIR=${1:?usage: $0 OVERLAY_DIR}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
BASE_OVERLAY=${BASE_OVERLAY:-$HOME/code/verl_docker/runtime_overlays/sglang_cudnn916_py310_arm64_notorch}
LIBERO_SRC_DIR=${LIBERO_SRC_DIR:-$HOME/code/verl_docker/runtime_overlays/libero_src/LIBERO}
mkdir -p "$OVERLAY_DIR" "$(dirname "$LIBERO_SRC_DIR")"

python3 -m pip install --target "$OVERLAY_DIR" --upgrade --no-cache-dir \
  'numpy<2' \
  'huggingface_hub<1.0' \
  'fsspec<=2025.10.0' \
  -r "$REPO_DIR/verl/experimental/vla/requirements_vla.txt" \
  mujoco==3.5.0 \
  robosuite==1.4.0 \
  bddl==1.0.1 \
  future==0.18.2 \
  cloudpickle==2.1.0 \
  orjson==3.11.3 \
  pyvers==0.1.0 \
  tensordict==0.10.0 \
  termcolor

find "$OVERLAY_DIR" -maxdepth 1 \( -name 'torch*' -o -name 'nvidia_*' -o -name 'triton*' \) -exec rm -rf {} +
if [[ -f "$OVERLAY_DIR/robosuite/macros.py" && ! -f "$OVERLAY_DIR/robosuite/macros_private.py" ]]; then
  cp "$OVERLAY_DIR/robosuite/macros.py" "$OVERLAY_DIR/robosuite/macros_private.py"
fi

if [[ ! -d "$LIBERO_SRC_DIR/.git" ]]; then
  rm -rf "$LIBERO_SRC_DIR"
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_SRC_DIR"
fi

PYTHONPATH="$LIBERO_SRC_DIR:$OVERLAY_DIR:$BASE_OVERLAY${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import importlib
mods = ["libero.libero", "mujoco", "robosuite", "bddl", "timm", "draccus", "json_numpy", "jsonlines"]
for mod in mods:
    importlib.import_module(mod)
    print(f"import_ok {mod}")
PY
