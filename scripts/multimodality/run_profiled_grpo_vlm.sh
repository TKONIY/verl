#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-sync_colocate}
MODEL_SIZE=${MODEL_SIZE:-7b}
ENGINE=${ENGINE:-sglang}
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-64}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-32}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-1}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4}
PROFILE_STEPS=${PROFILE_STEPS:-[1]}
GLOBAL_PROFILER_TOOL=${GLOBAL_PROFILER_TOOL:-nsys}
RUN_ROOT=${RUN_ROOT:-${PWD}/runs/multimodality}
RUN_NAME=${RUN_NAME:-${MODE}_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
TRAIN_FILES=${TRAIN_FILES:-$HOME/data/geo3k/train.parquet}
VAL_FILES=${VAL_FILES:-$HOME/data/geo3k/test.parquet}
EXTRA_ARGS=("$@")

TOTAL_GPUS=$((NNODES * N_GPUS_PER_NODE))
if (( TOTAL_GPUS >= 64 )); then
  echo "Refusing to launch with ${TOTAL_GPUS} GPUs; keep usage below 64 GPUs." >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
export VERL_FILE_LOGGER_PATH="${RUN_DIR}/metrics.jsonl"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

case "${MODEL_SIZE}" in
  3b)
    MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
    ROLLOUT_TP=${ROLLOUT_TP:-1}
    ACTOR_TP=${ACTOR_TP:-1}
    ACTOR_MICRO_BATCH=${ACTOR_MICRO_BATCH:-4}
    LOGPROB_MICRO_BATCH=${LOGPROB_MICRO_BATCH:-4}
    GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.65}
    ;;
  7b)
    MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}
    ROLLOUT_TP=${ROLLOUT_TP:-2}
    ACTOR_TP=${ACTOR_TP:-2}
    ACTOR_MICRO_BATCH=${ACTOR_MICRO_BATCH:-2}
    LOGPROB_MICRO_BATCH=${LOGPROB_MICRO_BATCH:-2}
    GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}
    ;;
  32b)
    MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-32B-Instruct}
    ROLLOUT_TP=${ROLLOUT_TP:-4}
    ACTOR_TP=${ACTOR_TP:-4}
    ACTOR_MICRO_BATCH=${ACTOR_MICRO_BATCH:-1}
    LOGPROB_MICRO_BATCH=${LOGPROB_MICRO_BATCH:-1}
    GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.45}
    ;;
  *)
    echo "Unsupported MODEL_SIZE=${MODEL_SIZE}. Use 3b, 7b, or 32b." >&2
    exit 1
    ;;
esac

TRAINER_GPU_COUNT=${TOTAL_GPUS}
if [[ "${MODE}" != "sync_colocate" ]]; then
  TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE:-2}
  TRAINER_GPU_COUNT=$((NNODES * TRAINER_N_GPUS_PER_NODE))
fi
if (( TRAINER_GPU_COUNT <= 0 )); then
  echo "Invalid TRAINER_GPU_COUNT=${TRAINER_GPU_COUNT}" >&2
  exit 1
fi
NORMALIZED_PPO_MINI_BATCH=$(( TRAIN_BATCH_SIZE / TRAINER_GPU_COUNT ))
if (( NORMALIZED_PPO_MINI_BATCH < 1 )); then
  echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} is too small for ${TRAINER_GPU_COUNT} trainer GPUs" >&2
  exit 1
fi
if (( ACTOR_MICRO_BATCH > NORMALIZED_PPO_MINI_BATCH )); then
  ACTOR_MICRO_BATCH=${NORMALIZED_PPO_MINI_BATCH}
fi
if (( LOGPROB_MICRO_BATCH > NORMALIZED_PPO_MINI_BATCH )); then
  LOGPROB_MICRO_BATCH=${NORMALIZED_PPO_MINI_BATCH}
fi

COMMON_ARGS=(
  algorithm.adv_estimator=grpo
  data.train_files="${TRAIN_FILES}"
  data.val_files="${VAL_FILES}"
  data.image_key=images
  data.max_prompt_length=1024
  data.max_response_length=2048
  data.filter_overlong_prompts=True
  data.truncation=error
  data.train_max_samples=${TRAIN_MAX_SAMPLES}
  data.val_max_samples=${VAL_MAX_SAMPLES}
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.01
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BATCH}
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH}
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
  actor_rollout_ref.rollout.name=${ENGINE}
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}
  actor_rollout_ref.rollout.enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL:-False}
  actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE:-True}
  actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER:-False}
  actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT}
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH}
  algorithm.use_kl_in_reward=False
  trainer.critic_warmup=0
  trainer.logger='["console","file"]'
  trainer.project_name=verl_multimodal_profile
  trainer.experiment_name="${RUN_NAME}"
  trainer.total_epochs=${TOTAL_EPOCHS}
  trainer.save_freq=-1
  trainer.test_freq=0
  trainer.val_before_train=False
  global_profiler.tool=${GLOBAL_PROFILER_TOOL}
  global_profiler.steps="${PROFILE_STEPS}"
  global_profiler.profile_continuous_steps=False
  global_profiler.save_path="${RUN_DIR}/profiles"
)

case "${MODE}" in
  sync_colocate)
    python3 -m verl.trainer.main_ppo \
      "${COMMON_ARGS[@]}" \
      data.train_batch_size=${TRAIN_BATCH_SIZE} \
      trainer.nnodes=${NNODES} \
      trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
      "${EXTRA_ARGS[@]}"
    ;;
  fully_async_disaggregate)
    TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE:-2}
    ROLLOUT_N_GPUS_PER_NODE=${ROLLOUT_N_GPUS_PER_NODE:-2}
    if (( TRAINER_N_GPUS_PER_NODE + ROLLOUT_N_GPUS_PER_NODE > N_GPUS_PER_NODE )); then
      echo "trainer + rollout GPUs per node exceed node capacity" >&2
      exit 1
    fi
    python3 -m verl.experimental.fully_async_policy.fully_async_main \
      --config-path=config \
      --config-name=fully_async_ppo_trainer.yaml \
      "${COMMON_ARGS[@]}" \
      actor_rollout_ref.hybrid_engine=False \
      data.train_batch_size=0 \
      data.gen_batch_size=${GEN_BATCH_SIZE} \
      data.return_raw_chat=True \
      trainer.nnodes=${NNODES} \
      trainer.n_gpus_per_node=${TRAINER_N_GPUS_PER_NODE} \
      rollout.nnodes=${NNODES} \
      rollout.n_gpus_per_node=${ROLLOUT_N_GPUS_PER_NODE} \
      rollout.n=${N_RESP_PER_PROMPT} \
      rollout.total_epochs=${TOTAL_EPOCHS} \
      rollout.total_rollout_steps=$(( TRAIN_MAX_SAMPLES * N_RESP_PER_PROMPT * TOTAL_EPOCHS )) \
      rollout.test_freq=0 \
      trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-1} \
      async_training.staleness_threshold=${STALENESS_THRESHOLD:-0.5} \
      async_training.trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP:-1} \
      async_training.require_batches=${REQUIRE_BATCHES:-1} \
      async_training.partial_rollout=${PARTIAL_ROLLOUT:-True} \
      "${EXTRA_ARGS[@]}"
    ;;
  one_step_off_disaggregate)
    TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE:-2}
    ROLLOUT_N_GPUS_PER_NODE=${ROLLOUT_N_GPUS_PER_NODE:-2}
    if (( TRAINER_N_GPUS_PER_NODE + ROLLOUT_N_GPUS_PER_NODE > N_GPUS_PER_NODE )); then
      echo "trainer + rollout GPUs per node exceed node capacity" >&2
      exit 1
    fi
    python3 -m verl.experimental.one_step_off_policy.main_ppo \
      --config-path=config \
      --config-name=one_step_off_ppo_trainer.yaml \
      "${COMMON_ARGS[@]}" \
      actor_rollout_ref.hybrid_engine=False \
      data.train_batch_size=${TRAIN_BATCH_SIZE} \
      data.return_raw_chat=True \
      trainer.nnodes=${NNODES} \
      trainer.n_gpus_per_node=${TRAINER_N_GPUS_PER_NODE} \
      rollout.nnodes=${NNODES} \
      rollout.n_gpus_per_node=${ROLLOUT_N_GPUS_PER_NODE} \
      actor_rollout_ref.rollout.mode=async \
      "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use sync_colocate, fully_async_disaggregate, or one_step_off_disaggregate." >&2
    exit 1
    ;;
esac

echo "Metrics JSONL: ${VERL_FILE_LOGGER_PATH}"
echo "Run report: python3 scripts/multimodality/summarize_profile.py --input ${VERL_FILE_LOGGER_PATH} --output-dir ${RUN_DIR}/report --label ${RUN_NAME}"
