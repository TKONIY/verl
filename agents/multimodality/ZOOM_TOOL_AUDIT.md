# Zoom Tool Audit

## Scope
- Ran `git submodule update --init --recursive` and confirmed that `recipe/` now includes `recipe/deepeyes/`.
- Searched `recipe/`, `examples/`, `verl/`, `tests/`, and `third_party/` for `deepeyes`, `zoom_in`, `zoom_out`, `image_zoom`, and related crop/search terms.

## Confirmed Runnable Example
- `recipe/deepeyes/` is the main visual tool-calling RL example in this repository.
- `recipe/deepeyes/run_deepeyes_grpo.sh` launches GRPO training and passes `actor_rollout_ref.rollout.multi_turn.tool_config_path=recipe/deepeyes/configs/image_zoom_in_tool_config.yaml`.
- `recipe/deepeyes/configs/image_zoom_in_tool_config.yaml` registers `verl.tools.image_zoom_in_tool.ImageZoomInTool` under the callable name `image_zoom_in_tool`.
- `recipe/deepeyes/deepeyes.py` contains the tool schema and prompt examples showing bbox-based zoom calls.

## Other References
- `verl/tools/image_zoom_in_tool.py` is the repository implementation of the bbox crop-and-return zoom tool.
- `tests/utils/dataset/test_multiturn_sft_dataset_on_cpu.py` contains a small unit-style example of an `image_zoom_in_tool` call and response.
- `examples/tutorial/agent_loop_get_started/agent_loop_tutorial.ipynb` mentions `deepeyes` as the canonical multi-turn visual tool example.

## What Is Not Present
- No other runnable `recipe/` or `examples/` workflow was found that uses `image_zoom_in_tool` or a separate `zoom_out` visual tool.
- No `zoom_out` tool implementation was found under `verl/tools/`.
- `third_party/cosmos-transfer2.5` contains `zoom_in` / `zoom_out` assets and camera-motion docs, but those are plenoptic/video controls rather than VLM tool-calling examples.
- `third_party/lingbot-vla/deploy/image_tools.py` and `third_party/lingbot-va/.../image_tools.py` provide resize/pad or uint8 conversion helpers for transport, not RL-time zoom-in/zoom-out tools.
