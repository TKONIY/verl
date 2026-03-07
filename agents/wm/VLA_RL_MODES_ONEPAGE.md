# verl 任务与训练模式一页速查

## 先看任务类型

```text
文本推理 RL
  ├─ 例子: GSM8K / MATH
  ├─ 输入/输出: 文本 -> 文本答案
  └─ 常见算法: GRPO, PPO

多模态 VLM RL
  ├─ 例子: Geo3K, Qwen2.5-VL
  ├─ 输入/输出: 图像+文本 -> 文本答案
  └─ 常见算法: GRPO

偏好对齐 RLHF
  ├─ 例子: HH-RLHF
  ├─ 输入/输出: Prompt -> 偏好更好的回复
  └─ 常见算法: PPO + reward model + critic

机器人 VLA RL
  ├─ 例子: LIBERO, IsaacSim, Mujoco
  ├─ 输入/输出: 视觉/语言/状态 -> 动作序列
  └─ 常见算法: VLA-GRPO, SAC

SFT
  ├─ 例子: VLM supervised fine-tuning
  ├─ 输入/输出: 标注样本 -> 监督学习
  └─ 常见算法: 无 RL，仅 SFT
```

## 再看训练模式

| 模式 | 适合任务 | Critic | Reward 来源 | On/Off-policy | 关键词 |
| --- | --- | --- | --- | --- | --- |
| `PPO` | RLHF、偏好对齐 | 需要 | reward model / 规则 | On-policy | 稳定、标准 actor-critic |
| `GRPO` | 数学、代码、可验证推理 | 不需要 | 组内相对奖励 / 规则 | On-policy | 多采样、组归一化 |
| `VLA-GRPO` | 机器人策略优化 | 通常不需要 | 环境奖励 | On-policy | 环境交互、异步 env loop |
| `SAC` | 连续控制、样本昂贵环境 | Q/value 结构 | 环境奖励 | Off-policy | replay buffer、样本复用 |
| `SFT` | 初始化、模仿学习 | 不需要 | 无 | 不适用 | 先学会再进 RL |

## 这个仓库里最值得先看的入口

| 目标 | 先看哪个文件 |
| --- | --- |
| 理解文本可验证 RL | `examples/grpo_trainer/run_qwen2-7b.sh` |
| 理解 PPO + RLHF | `examples/ppo_trainer/run_deepseek_full_hh_rlhf.sh` |
| 理解多模态 RL | `examples/grpo_trainer/run_qwen2_5_vl-7b.sh` |
| 理解 VLA 系统设计 | `verl/experimental/vla/README.md` |
| 理解 VLA-GRPO 入口 | `verl/experimental/vla/run_simpleVLA_libero_grpo.sh` |
| 理解 VLA-SAC 入口 | `verl/experimental/vla/run_pi05_libero_sac.sh` |

## 最短判断规则

- 有 `critic`、有 `reward model`：大概率是 `PPO / RLHF`。
- 没 `critic`、同一题采样多次：大概率是 `GRPO`。
- 需要 simulator / environment step：这是 `VLA RL`。
- 有 `replay buffer`、强调历史样本复用：这是 `SAC`。
- 没有 rollout、只有监督数据：这是 `SFT`。

## 一句话版本

- 文本/图文可验证任务，优先看 `GRPO`。
- 偏好对齐任务，优先看 `PPO`。
- 机器人控制任务，重点看 `VLA-GRPO` 和 `SAC`。
- `VLA` 比普通 LLM RL 多了真实环境交互与调度问题。
