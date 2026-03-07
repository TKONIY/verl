# Image Transfer Summary

## Short Answer
- 跨机器时，不是“只传 metadata 就够了”，但也不应该默认直接传原始图片字节。
- 推荐在 replay / queue 里主要存 `image path`、`cache key`、`sample id`、token ids 和 PPO 标量信息。
- 真正需要做视觉 forward 的 worker 再本地读图、decode，或在少数情况下接收已经物化的视觉张量。

## When Images Are Needed
- `rollout` 生成前：rollout worker 必须能拿到图片或其等价视觉输入。
- `ref` / `old-log-prob` 重算时：对应 worker 也必须重新拿到同一张图或可替代表示。
- `reward` 只有在依赖视觉输入时才需要再次拿图。

## Who Sends What
- 常见路径是 `shared storage -> worker`：worker 收到图片引用后自行读图。
- 在 `async + disaggregate` 里，`queue / replay` 更适合传 `metadata`，不适合长期塞原始图片字节。
- 只有当重复 decode 明显更贵时，才考虑跨机器传 `pixel_values` 或视觉 embedding。

## Practical Takeaway
- 单节点时，图片相关开销通常不是主瓶颈，生成阶段更重。
- 到 `32 GPU` 时，多模态 payload 和同步成本已经明显升高。
- 当前实验里更像是“共享存储读取 + decode + 同步”共同形成成本，而不是单一的“原图网络传输”。

## Current Evidence
- 定量占比见 `runs/multimodality/component_breakdown.md`。
- 总结报告见 `agents/multimodality/FINAL_REPORT.md`。
