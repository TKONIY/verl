# `verl-agent` Code Inspection Notes and Follow-up Plan

## Current Live Audit Result
A live inspection was run against `https://github.com/langfengQ/verl-agent` at local clone `/tmp/verl-agent-inspect` on commit `796ed31`.

## What Is Confirmed
- `verl-agent` does train VLM agents. Public README and example scripts explicitly mention `Qwen2.5-VL` and `Qwen3-VL`.
- The visible VLM training examples are environment-driven rather than tool-driven:
  - `examples/gigpo_trainer/run_sokoban.sh`
  - `examples/gigpo_trainer/run_sokoban_qwen3vl.sh`
  - related tasks include `EZPoints` and `NumberLine`
- The VLM path uses visual observations from environments and normal VLM inputs such as `data.image_key=images`, rather than a separate image crop / zoom tool.
- A `Search` environment exists, with tool-like search code under `agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py`.

## What Is Not Confirmed / Currently Looks Absent
- No repo-level evidence was found for a dedicated image crop / zoom tool comparable to `verl/tools/image_zoom_in_tool.py`.
- No repo-level evidence was found for image search / visual retrieval / reverse-image-search tooling.
- The visible `search` capability appears to be text retrieval / search API access, not image search.

## Evidence Table
| Component | Exists | Evidence | Conclusion |
| --- | --- | --- | --- |
| VLM training | Yes | `README.md`, `examples/gigpo_trainer/run_sokoban.sh`, `examples/gigpo_trainer/run_sokoban_qwen3vl.sh` | VLM support is real and actively documented |
| Visual environment tasks | Yes | `Sokoban`, `EZPoints`, `NumberLine` examples in `README.md` | Main VLM usage is environment-driven |
| Search capability | Yes | `agent_system/environments/env_package/search/.../tools/search.py` | Search exists, but as text retrieval |
| Image crop / zoom tool | Not found | keyword audit over `crop`, `zoom`, `bbox`, `image_zoom` found no agent-facing VLM tool | Likely absent in current public repo |
| Image search / visual retrieval | Not found | keyword audit over `visual retrieval`, `image retrieval`, `reverse image` found no agent-facing implementation | Likely absent in current public repo |

## Practical Interpretation
- `verl-agent` is best understood as an **environment-driven VLM RL framework**.
- It is not currently a strong public example of **tool-centric agentic VLM** with crop / zoom or image-search tools.
- For a `verl`-native agentic VLM extension, the most likely hybrid design is:
  - environment-driven VLM rollout logic from `verl-agent`
  - tool-driven visual utilities from local `verl` such as `image_zoom_in_tool`

## If We Need a Deeper Second Pass
Inspect these paths in the cloned repo next:
- `examples/gigpo_trainer/`
- `agent_system/environments/env_package/`
- `verl/tools/`
- `examples/data_preprocess/`

Use this keyword set:
- `Qwen2.5-VL`, `Qwen3-VL`, `pixel_values`, `crop`, `zoom`, `bbox`, `grounding`, `search`, `retrieval`, `image retrieval`, `visual retrieval`, `OCR`

Expected final decision labels:
- `environment-driven`
- `tool-driven`
- `hybrid`

Current label: `environment-driven`
