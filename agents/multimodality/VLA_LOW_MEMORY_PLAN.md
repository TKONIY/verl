# VLA Low-Memory Runtime Plan

## Goal
Get the Libero GRPO smoke path to emit the first training step and profiler output without modifying core `verl` source.

## Observed Blocker
- `vla_libero_grpo_smoke10_wt_20260307_231332` reaches `ray.init` then hits Slurm OOM.
- `vla_libero_grpo_smoke11_wt_20260307_231610` avoids immediate OOM but stalls after `ray.init`; `sstat` shows `MaxRSS` around `110 GB` before any first-step metrics.

## Recommended Next Configuration Ladder
1. **Reduce actor / env replication further**
   - `N_ENV_GPUS_PER_NODE=1`
   - `ROLLOUT_N=1`
   - `TRAIN_BATCH_SIZE=1`
   - `VAL_BATCH_SIZE=1`
   - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`
   - `actor_rollout_ref.rollout.log_prob_micro_batch_size=1`
   - `actor_rollout_ref.ref.log_prob_micro_batch_size=1`
2. **Lower model-side memory pressure**
   - keep `enable_gradient_checkpointing=False` only if startup fragmentation is lower; otherwise test `True`
   - set stricter rollout memory budget, e.g. `actor_rollout_ref.rollout.gpu_memory_utilization=0.5`
   - keep only a single rollout sample and single env GPU
3. **Shrink control-plane concurrency**
   - pin `+ray_kwargs.ray_init.num_cpus` to a smaller value
   - avoid extra env workers or validation workers
4. **If still stalled after `ray.init`**
   - run the same script with a larger host-memory allocation, e.g. `--mem=320G` or above
   - if memory stops growing but first-step logs still never appear, inspect Ray actor creation and simulator boot timing separately from PPO

## Minimum Next Trial
```bash
RUN_NAME=vla_libero_grpo_smoke12 \
MEM=320G \
N_GPUS_PER_NODE=4 \
N_ENV_GPUS_PER_NODE=1 \
ROLLOUT_N=1 \
TRAIN_BATCH_SIZE=1 \
VAL_BATCH_SIZE=1 \
TOTAL_TRAINING_STEPS=1 \
TIME_LIMIT=02:00:00 \
bash scripts/vla/submit_profile_libero_grpo_slurm.sh
```

## Success Criterion
A successful next milestone is not full convergence. It is simply:
- first `step:` metrics line in `train.log`
- profiler output directory created
- no runaway RSS growth before the first step
