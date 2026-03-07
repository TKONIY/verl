#!/usr/bin/env bash
set -euo pipefail
MODE=sync_colocate exec "$(dirname "$0")/run_profiled_grpo_vlm.sh" "$@"
