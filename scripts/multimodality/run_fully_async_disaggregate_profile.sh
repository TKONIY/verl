#!/usr/bin/env bash
set -euo pipefail
MODE=fully_async_disaggregate exec "$(dirname "$0")/run_profiled_grpo_vlm.sh" "$@"
