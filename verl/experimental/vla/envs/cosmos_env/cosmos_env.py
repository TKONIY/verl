import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as tvf

from verl.experimental.vla.envs.action_utils import put_info_on_image, save_rollout_video, tile_images, to_tensor

logger = logging.getLogger(__name__)


def cfg_get(cfg, *names, default=None):
    for name in names:
        if name in cfg and cfg.get(name) is not None:
            return cfg.get(name)
    return default

DEFAULT_TASK_DESCRIPTIONS = [
    'move the block to the goal area',
    'align the gripper with the object',
    'push the object to the marked target',
    'place the item near the highlighted bin',
]


@dataclass
class BackendStepOutput:
    frame: np.ndarray
    state: np.ndarray


class BaseCosmosBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state_action_scale = float(cfg.get('state_action_scale', 0.08))

    def step(self, frame: np.ndarray, state: np.ndarray, action: np.ndarray, task: str, step_idx: int) -> BackendStepOutput:
        raise NotImplementedError

    def close(self) -> None:
        return


class MockCosmosBackend(BaseCosmosBackend):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.height = int(cfg_get(cfg, 'frame_height', 'image_height', default=224))
        self.width = int(cfg_get(cfg, 'frame_width', 'image_width', default=224))

    def step(self, frame: np.ndarray, state: np.ndarray, action: np.ndarray, task: str, step_idx: int) -> BackendStepOutput:
        del frame, task
        next_state = np.clip(state + self.state_action_scale * action[: state.shape[0]], -1.0, 1.0)
        next_frame = render_state_frame(
            state=next_state,
            task_bucket=step_idx % 4,
            height=self.height,
            width=self.width,
            step_idx=step_idx,
        )
        return BackendStepOutput(frame=next_frame, state=next_state)


class Predict2ActionConditionedBackend(BaseCosmosBackend):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.repo_root = Path(os.path.expanduser(str(cfg.get('repo_root', 'third_party/cosmos-predict2.5')))).resolve()
        self.checkpoint_path = str(cfg.get('model_name_or_path', ''))
        self.experiment_name = str(cfg.get('experiment_name', 'cosmos_predict2_action_conditioned'))
        self.config_file = str(
            cfg.get('config_file', 'cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py')
        )
        self.guidance = float(cfg.get('guidance', 7.0))
        self.num_steps = int(cfg.get('num_steps', 35))
        self.resolution = str(cfg.get('resolution', '256,320'))
        self.negative_prompt = str(cfg.get('negative_prompt', ''))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._predictor = None
        self._cleanup_environment = None
        self._frame_height = int(cfg_get(cfg, 'frame_height', 'image_height', default=224))
        self._frame_width = int(cfg_get(cfg, 'frame_width', 'image_width', default=224))

    def _ensure_model(self):
        if self._predictor is not None:
            return
        if not self.repo_root.exists():
            raise RuntimeError(
                f'Cosmos repo_root not found: {self.repo_root}. Initialize submodules and install Cosmos dependencies first.'
            )
        if not self.checkpoint_path:
            raise RuntimeError('env.train.cosmos.model_name_or_path must be set when backend=predict2')
        sys.path.insert(0, str(self.repo_root))
        try:
            from cosmos_oss.init import init_environment
            from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
        except Exception as exc:
            raise RuntimeError(
                'Failed to import cosmos-predict2.5 runtime. Install its CUDA extras and Python dependencies before '
                'using backend=predict2.'
            ) from exc

        init_environment()
        self._predictor = Video2WorldInference(
            experiment_name=self.experiment_name,
            ckpt_path=self.checkpoint_path,
            s3_credential_path='',
            context_parallel_size=1,
            config_file=self.config_file,
        )

    def step(self, frame: np.ndarray, state: np.ndarray, action: np.ndarray, task: str, step_idx: int) -> BackendStepOutput:
        self._ensure_model()
        input_frame = tvf.to_tensor(frame).unsqueeze(0)
        video_input = torch.cat([input_frame, torch.zeros_like(input_frame)], dim=0)
        video_input = (video_input * 255.0).to(torch.uint8).unsqueeze(0).permute(0, 2, 1, 3, 4)
        action_tensor = torch.from_numpy(action[None]).float()
        with torch.no_grad():
            video = self._predictor.generate_vid2world(
                prompt=task,
                input_path=video_input,
                action=action_tensor,
                guidance=self.guidance,
                num_video_frames=2,
                num_latent_conditional_frames=1,
                resolution=self.resolution,
                seed=step_idx,
                negative_prompt=self.negative_prompt,
                num_steps=self.num_steps,
            )
        video_normalized = (video - (-1)) / 2.0
        video_uint8 = (torch.clamp(video_normalized[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        next_frame = video_uint8[-1]
        if next_frame.shape[0] != self._frame_height or next_frame.shape[1] != self._frame_width:
            next_frame = np.array(
                tvf.resize(torch.from_numpy(next_frame).permute(2, 0, 1), [self._frame_height, self._frame_width])
                .permute(1, 2, 0)
                .to(torch.uint8)
            )
        next_state = np.clip(state + self.state_action_scale * action[: state.shape[0]], -1.0, 1.0)
        return BackendStepOutput(frame=next_frame, state=next_state)

    def close(self) -> None:
        if self._predictor is not None:
            self._predictor.cleanup()
            self._predictor = None


class CosmosEnv(gym.Env):
    def __init__(self, cfg, rank, world_size):
        self.rank = rank
        self.cfg = cfg
        self.world_size = world_size
        self.seed = int(self.cfg.seed) + rank
        self.num_envs = int(self.cfg.num_envs)
        self.action_dim = int(self.cfg.get('action_dim', 7))
        self.state_dim = int(cfg_get(self.cfg.get('cosmos', {}), 'state_dim', default=self.action_dim))
        self.video_cfg = self.cfg.video_cfg
        self.max_episode_steps = int(self.cfg.max_episode_steps)
        self._generator = np.random.default_rng(seed=self.seed)
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_step_reward = np.zeros(self.num_envs, dtype=np.float32)
        self.render_images = []
        self.video_cnt = 0

        self.cosmos_cfg = self.cfg.get('cosmos', {})
        self.total_states_per_task = int(cfg_get(self.cosmos_cfg, 'total_states_per_task', 'initial_state_count', default=64))
        self.goal_threshold = float(cfg_get(self.cosmos_cfg, 'goal_threshold', 'success_tolerance', default=0.18))
        self.success_reward = float(self.cosmos_cfg.get('success_reward', 2.5))
        self.step_penalty = float(self.cosmos_cfg.get('step_penalty', 0.01))
        self.action_penalty = float(self.cosmos_cfg.get('action_penalty', 0.02))
        self.frame_height = int(cfg_get(self.cosmos_cfg, 'frame_height', 'image_height', default=224))
        self.frame_width = int(cfg_get(self.cosmos_cfg, 'frame_width', 'image_width', default=224))
        tasks = cfg_get(self.cosmos_cfg, 'task_descriptions', 'task_catalog', default=DEFAULT_TASK_DESCRIPTIONS)
        self.task_catalog = list(tasks)
        self.num_tasks = len(self.task_catalog)
        self.goal_states = np.stack([self._build_goal_state(task_id) for task_id in range(self.num_tasks)], axis=0)

        self.current_state_ids = np.zeros(self.num_envs, dtype=np.int32)
        self.current_task_ids = np.zeros(self.num_envs, dtype=np.int32)
        self.current_states = np.zeros((self.num_envs, self.state_dim), dtype=np.float32)
        self.current_frames = np.zeros((self.num_envs, self.frame_height, self.frame_width, 3), dtype=np.uint8)
        self.prev_goal_distance = np.zeros(self.num_envs, dtype=np.float32)

        backend_name = str(self.cosmos_cfg.get('backend', 'mock')).lower()
        if backend_name in {'predict2', 'cosmos_predict2', 'cosmos_predict2_stub'}:
            self.backend = Predict2ActionConditionedBackend(self.cosmos_cfg)
        elif backend_name == 'mock':
            self.backend = MockCosmosBackend(self.cosmos_cfg)
        else:
            raise ValueError(f'Unsupported cosmos backend: {backend_name}')

        self._init_metrics()

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def _build_goal_state(self, task_id: int) -> np.ndarray:
        rng = np.random.default_rng(seed=10_000 + task_id)
        return rng.uniform(low=-0.75, high=0.75, size=(self.state_dim,)).astype(np.float32)

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)

    def _reset_metrics(self, env_idx=None):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        env_idx = np.asarray(env_idx)
        self.prev_step_reward[env_idx] = 0.0
        self.success_once[env_idx] = False
        self.returns[env_idx] = 0.0
        self._elapsed_steps[env_idx] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | np.asarray(terminations)
        episode_info['success_once'] = self.success_once.copy()
        episode_info['return'] = self.returns.copy()
        episode_info['episode_len'] = self.elapsed_steps.copy()
        safe_steps = np.maximum(self.elapsed_steps, 1)
        episode_info['reward'] = episode_info['return'] / safe_steps
        infos['episode'] = to_tensor(episode_info)
        return infos

    def get_all_state_ids(self):
        return np.arange(self.total_states_per_task * self.num_tasks)

    def _seeded_initial_state(self, state_id: int, task_id: int) -> np.ndarray:
        rng = np.random.default_rng(seed=self.seed + task_id * 10_000 + state_id)
        return rng.uniform(low=-0.5, high=0.5, size=(self.state_dim,)).astype(np.float32)

    def _task_descriptions_for_current(self) -> list[str]:
        return [self.task_catalog[task_id % self.num_tasks] for task_id in self.current_task_ids]

    def _wrap_current_obs(self):
        images_and_states = {
            'full_image': self.current_frames.copy(),
            'wrist_image': self.current_frames.copy(),
            'state': self.current_states.copy(),
        }
        return {
            'images_and_states': to_tensor(images_and_states),
            'task_descriptions': self._task_descriptions_for_current(),
        }

    def reset(self, env_idx: Optional[int | list[int] | np.ndarray] = None, reset_state_ids=None, options: Optional[dict] = None):
        del options
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        env_idx = np.asarray(env_idx)
        if reset_state_ids is None:
            reset_state_ids = self._generator.integers(0, self.total_states_per_task, size=len(env_idx))
        for local_pos, env_id in enumerate(env_idx):
            state_id = int(reset_state_ids[local_pos])
            task_id = int(self.current_task_ids[env_id]) if self.num_tasks > 0 else 0
            self.current_state_ids[env_id] = state_id
            self.current_states[env_id] = self._seeded_initial_state(state_id, task_id)
            self.current_frames[env_id] = render_state_frame(
                state=self.current_states[env_id],
                task_bucket=task_id,
                height=self.frame_height,
                width=self.frame_width,
                step_idx=0,
            )
            self.prev_goal_distance[env_id] = np.linalg.norm(self.current_states[env_id] - self.goal_states[task_id])
        self._reset_metrics(env_idx)
        return self._wrap_current_obs(), {}

    def step(self, actions=None):
        if actions is None:
            obs, infos = self.reset(reset_state_ids=self.current_state_ids)
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)
            return obs, None, to_tensor(terminations), to_tensor(truncations), infos

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        self._elapsed_steps += 1

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = self._elapsed_steps >= self.max_episode_steps
        infos = {}

        for env_id in range(self.num_envs):
            task_id = int(self.current_task_ids[env_id] % self.num_tasks)
            task_description = self.task_catalog[task_id]
            output = self.backend.step(
                frame=self.current_frames[env_id],
                state=self.current_states[env_id],
                action=actions[env_id],
                task=task_description,
                step_idx=int(self._elapsed_steps[env_id]),
            )
            self.current_frames[env_id] = output.frame
            self.current_states[env_id] = output.state
            goal_distance = np.linalg.norm(output.state - self.goal_states[task_id])
            progress = self.prev_goal_distance[env_id] - goal_distance
            reward = progress - self.step_penalty - self.action_penalty * float(np.linalg.norm(actions[env_id]))
            terminated = goal_distance < self.goal_threshold
            if terminated:
                reward += self.success_reward
            rewards[env_id] = reward
            terminations[env_id] = terminated
            self.prev_goal_distance[env_id] = goal_distance

        obs = self._wrap_current_obs()
        if self.video_cfg.save_video:
            plot_infos = {
                'rewards': rewards,
                'terminations': terminations,
                'task': np.array(self._task_descriptions_for_current()),
            }
            self.add_new_frames(obs, plot_infos)

        infos = self._record_metrics(rewards, terminations, infos)
        return obs, to_tensor(rewards), to_tensor(terminations), to_tensor(truncations), infos

    def chunk_step(self, chunk_actions):
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        chunk_actions = np.asarray(chunk_actions, dtype=np.float32)
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        extracted_obs = None
        infos = {}
        for chunk_idx in range(chunk_size):
            extracted_obs, step_reward, terminations, truncations, infos = self.step(chunk_actions[:, chunk_idx])
            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)
        return (
            extracted_obs,
            torch.stack(chunk_rewards, dim=1),
            torch.stack(raw_chunk_terminations, dim=1),
            torch.stack(raw_chunk_truncations, dim=1),
            infos,
        )

    def add_new_frames(self, obs, plot_infos):
        images = []
        for env_id, img in enumerate(obs['images_and_states']['full_image']):
            info_item = {k: v if np.size(v) == 1 else v[env_id] for k, v in plot_infos.items()}
            images.append(put_info_on_image(img.cpu().numpy(), info_item))
        self.render_images.append(tile_images(images, nrows=max(1, int(np.sqrt(self.num_envs)))))

    def flush_video(self, video_sub_dir: Optional[str] = None):
        output_dir = os.path.join(self.video_cfg.video_base_dir, f'rank_{self.rank}')
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, video_sub_dir)
        save_rollout_video(self.render_images, output_dir=output_dir, video_name=f'{self.video_cnt}')
        self.video_cnt += 1
        self.render_images = []

    def close(self):
        self.backend.close()

    def load_state(self, state_buffer: bytes):
        del state_buffer
        return None

    def get_state(self):
        return None

    def reset_envs_to_state_ids(self, state_ids_list, task_ids_list):
        env_idx = np.arange(len(state_ids_list))
        self.current_state_ids[: len(state_ids_list)] = np.asarray(state_ids_list, dtype=np.int32)
        self.current_task_ids[: len(task_ids_list)] = np.asarray(task_ids_list, dtype=np.int32) % self.num_tasks
        obs, infos = self.reset(env_idx=env_idx, reset_state_ids=state_ids_list)
        return obs, infos


def render_state_frame(state: np.ndarray, task_bucket: int, height: int, width: int, step_idx: int) -> np.ndarray:
    grid_y, grid_x = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing='ij',
    )
    phase = (task_bucket % 4) / 4.0
    state = np.pad(state.astype(np.float32), (0, max(0, 3 - len(state))), mode='wrap')
    red = np.mod(grid_x + 0.35 * state[0] + 0.03 * step_idx + phase, 1.0)
    green = np.mod(grid_y + 0.35 * state[1] + 0.02 * step_idx + phase / 2.0, 1.0)
    blue = np.mod((grid_x + grid_y) / 2.0 + 0.35 * state[2] + 0.01 * step_idx + phase / 3.0, 1.0)
    frame = np.stack([red, green, blue], axis=-1)
    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)
