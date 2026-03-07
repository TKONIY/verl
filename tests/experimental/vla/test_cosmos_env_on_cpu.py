import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class AttrDict(dict):
    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[key] = value
        return value


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _install_action_utils_stub():
    module = types.ModuleType("verl.experimental.vla.envs.action_utils")

    def to_tensor(array, device="cpu"):
        if isinstance(array, dict):
            return {key: to_tensor(value, device=device) for key, value in array.items()}
        if isinstance(array, torch.Tensor):
            result = array.to(device)
        else:
            result = torch.as_tensor(array, device=device)
        if result.dtype == torch.float64:
            result = result.to(torch.float32)
        return result

    def tile_images(images, nrows=1):
        del nrows
        return np.concatenate(images, axis=1)

    def put_info_on_image(image, info, extras=None, overlay=True):
        del info, extras, overlay
        return image

    def save_rollout_video(rollout_images, output_dir, video_name, fps=30):
        del rollout_images, fps
        os.makedirs(output_dir, exist_ok=True)
        with open(Path(output_dir) / f"{video_name}.mp4", "wb") as file_obj:
            file_obj.write(b"stub")

    module.to_tensor = to_tensor
    module.tile_images = tile_images
    module.put_info_on_image = put_info_on_image
    module.save_rollout_video = save_rollout_video
    sys.modules[module.__name__] = module


def _load_cosmos_env_class():
    repo_root = Path(__file__).resolve().parents[3]
    cosmos_env_path = repo_root / "verl" / "experimental" / "vla" / "envs" / "cosmos_env" / "cosmos_env.py"

    sys.modules.setdefault("verl", types.ModuleType("verl"))
    sys.modules.setdefault("verl.experimental", types.ModuleType("verl.experimental"))
    sys.modules.setdefault("verl.experimental.vla", types.ModuleType("verl.experimental.vla"))
    sys.modules.setdefault("verl.experimental.vla.envs", types.ModuleType("verl.experimental.vla.envs"))
    _install_action_utils_stub()
    cosmos_module = _load_module("verl.experimental.vla.envs.cosmos_env.cosmos_env", cosmos_env_path)
    return cosmos_module.CosmosEnv


def test_cosmos_env_creation_and_step_on_cpu(tmp_path):
    cosmos_env_cls = _load_cosmos_env_class()
    num_envs = 4
    cfg = AttrDict(
        {
            "max_episode_steps": 8,
            "only_eval": False,
            "reward_coef": 1.0,
            "video_cfg": {
                "save_video": True,
                "video_base_dir": str(tmp_path),
            },
            "num_envs": num_envs,
            "seed": 0,
            "cosmos": {
                "backend": "mock",
                "initial_state_count": 16,
                "image_height": 96,
                "image_width": 96,
                "task_catalog": [
                    "Move to the green marker.",
                    "Move and close the gripper.",
                ],
            },
        }
    )

    env = cosmos_env_cls(cfg, rank=0, world_size=1)
    obs, infos = env.reset_envs_to_state_ids([0, 1, 2, 3], [0, 1, 0, 1])
    assert infos == {}
    assert obs["images_and_states"]["full_image"].shape == (num_envs, 96, 96, 3)
    assert obs["images_and_states"]["wrist_image"].shape == (num_envs, 96, 96, 3)
    assert obs["images_and_states"]["state"].shape == (num_envs, 7)
    assert len(obs["task_descriptions"]) == num_envs

    actions = np.zeros((num_envs, 3, 7), dtype=np.float32)
    actions[:, :, 0] = 0.5
    next_obs, rewards, terminations, truncations, infos = env.chunk_step(actions)
    assert next_obs["images_and_states"]["full_image"].shape == (num_envs, 96, 96, 3)
    assert rewards.shape == (num_envs, 3)
    assert terminations.shape == (num_envs, 3)
    assert truncations.shape == (num_envs, 3)
    assert "episode" in infos

    env.flush_video(video_sub_dir="smoke")
    video_path = tmp_path / "rank_0" / "smoke" / "0.mp4"
    assert os.path.exists(video_path)
    env.close()
