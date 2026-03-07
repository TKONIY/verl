set -euo pipefail
set -x

cosmos_data_dir=${COSMOS_DATA_DIR:-"$HOME/data/cosmos_robot_rl"}
cosmos_train_path=${COSMOS_TRAIN_PATH:-"$cosmos_data_dir/train.parquet"}
cosmos_test_path=${COSMOS_TEST_PATH:-"$cosmos_data_dir/test.parquet"}
train_files=$cosmos_train_path
test_files=$cosmos_test_path

OUTPUT_DIR=${MLP_MODEL_OUTPUT:-"$HOME/models/vla_cosmos_sac"}
VIDEO_OUTPUT=${VIDEO_OUTPUT:-"$HOME/video/cosmos_sac"}
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"$HOME/data/pi05_libero_torch"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"$SFT_MODEL_PATH"}
COSMOS_REPO_ROOT=${COSMOS_REPO_ROOT:-"$PWD/third_party/cosmos-predict2.5"}
COSMOS_MODEL_PATH=${COSMOS_MODEL_PATH:-""}
COSMOS_BACKEND=${COSMOS_BACKEND:-"mock"}

NUM_NODES=${NUM_NODES:-1}
NUM_GPUS=${NUM_GPUS:-4}
NUM_ENV_GPUS=${NUM_ENV_GPUS:-2}
NUM_ROLLOUT_GPUS=${NUM_ROLLOUT_GPUS:-$((NUM_GPUS - NUM_ENV_GPUS))}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-1}
NUM_STAGE=${NUM_STAGE:-1}
NUM_ENV=${NUM_ENV:-4}
NUM_ACTION_CHUNKS=${NUM_ACTION_CHUNKS:-4}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-64}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
PROJECT_NAME=${PROJECT_NAME:-"vla_cosmos_rl"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"cosmos_sac_single_node"}

if [ ! -f "$train_files" ] || [ ! -f "$test_files" ]; then
    python -m verl.experimental.vla.prepare_cosmos_dataset --local_save_dir "$cosmos_data_dir"
fi

python -m verl.experimental.vla.main_sac \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=4 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    env.rollout.pipeline_stage_num=$NUM_STAGE \
    env.train.simulator_type=cosmos \
    env.train.num_envs=$NUM_ENV \
    env.train.max_episode_steps=$MAX_EPISODE_STEPS \
    env.train.only_eval=False \
    env.train.seed=42 \
    env.train.video_cfg.save_video=True \
    env.train.video_cfg.video_base_dir=${VIDEO_OUTPUT} \
    env.train.cosmos.backend=${COSMOS_BACKEND} \
    env.train.cosmos.repo_root=${COSMOS_REPO_ROOT} \
    env.train.cosmos.model_name_or_path=${COSMOS_MODEL_PATH} \
    env.train.cosmos.total_states_per_task=64 \
    env.train.cosmos.frame_height=224 \
    env.train.cosmos.frame_width=224 \
    env.train.cosmos.state_dim=7 \
    env.train.cosmos.batch_inference_size=1 \
    env.actor.model.num_action_chunks=$NUM_ACTION_CHUNKS \
    env.actor.model.action_dim=7 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.tokenizer_path=$TOKENIZER_PATH \
    actor_rollout_ref.rollout.mode=async_envloop \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=False \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    +actor_rollout_ref.algorithm=sac \
    trainer.logger=[console] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_GPUS \
    +trainer.n_env_gpus_per_node=$NUM_ENV_GPUS \
    +trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=30 \
    trainer.test_freq=-1 \
    trainer.total_epochs=5 \
    trainer.val_only=False \
    trainer.val_before_train=False
