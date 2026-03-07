from __future__ import annotations

from typing import Any

import torch

from verl.trainer.ppo import reward as reward_mod


def _compat_compute_reward(data: Any, reward_fn: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    except Exception:
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}
    return reward_tensor, reward_extra_infos_dict


if not hasattr(reward_mod, "compute_reward"):
    reward_mod.compute_reward = _compat_compute_reward
