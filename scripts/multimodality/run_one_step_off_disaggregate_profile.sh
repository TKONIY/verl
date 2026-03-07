#!/usr/bin/env bash
set -euo pipefail
MODE=one_step_off_disaggregate exec "$(dirname "$0")/run_profiled_grpo_vlm.sh" "$@"
