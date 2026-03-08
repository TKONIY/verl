# Agentic VLM Open-Source Comparison

## Comparison Table
| Project | URL | Main task family | VLM | RL | Driver style | Crop / Zoom | Image search / visual retrieval | Fit for current work |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `verl` multiturn + agent loop | Local repo | multiturn VLM RL, tool loop, dataset-driven | Yes | Yes | hybrid-capable, current examples mostly dataset/tool validation | Yes, via `image_zoom_in_tool` | Text search exists; image search not found | Best local base for extensions |
| `verl-agent` | https://github.com/langfengQ/verl-agent | environment-driven VLM agents, visual games, search env | Yes | Yes | environment-driven | Not found in public repo audit | Not found; search appears text-only | Best external reference for `verl`-adjacent VLM RL |
| `CogAgent` | https://github.com/zai-org/CogAgent | GUI / desktop agent | Yes | Not primarily | tool / action agent | GUI grounding oriented, crop support not confirmed here | Not a focus | Strong GUI-agent reference |
| `Aguvis` | https://github.com/xlang-ai/aguvis | pure-vision GUI agent | Yes | Not primarily | environment / GUI interaction | Vision-grounding oriented, crop support task-dependent | Not a focus | Strong OSWorld-style benchmark reference |
| `MobileAgent` | https://github.com/X-PLUG/MobileAgent | mobile GUI agent | Yes | Usually imitation/agent first | environment / GUI interaction | Likely visual grounding, not confirmed here as standalone crop tool | Not a focus | Strong mobile-agent reference |
| `AppAgent` | https://github.com/TencentQQGYLab/AppAgent | smartphone app agent | Yes | Not primarily | environment / GUI interaction | Not confirmed here as standalone crop tool | Not a focus | Good mobile interaction reference |
| `Open-AgentRL` | https://github.com/Gen-Verse/Open-AgentRL | general agentic RL | Mixed | Yes | RL-framework first | Not the focus | Not the focus | Strong RL design reference |

## Ranked Recommendations
### Best for extending current `verl` work
1. `verl-agent`
2. local `verl` multiturn + `image_zoom_in_tool`
3. `Open-AgentRL`

### Best for GUI / OS benchmark ideas
1. `Aguvis`
2. `CogAgent`
3. `MobileAgent` / `AppAgent`

### Best for 32-GPU distributed profiling inspiration
1. `verl-agent`
2. local `verl`
3. `Open-AgentRL`

## Notes
- The local `verl` repo already contains a crop-capable visual tool: `verl/tools/image_zoom_in_tool.py`.
- The local `verl` repo also contains text search tools, but no confirmed image-search / visual-retrieval tool was found in the current audit.
- `verl-agent` is the most relevant external project because it is both `verl`-adjacent and explicitly supports `Qwen2.5-VL` / `Qwen3-VL`.
