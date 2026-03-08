# 安装与仓库接入规范

## 范围

这份说明记录新的仓库根目录 `third_party/` 的接入和安装规范。

## 已纳入仓库

以下仓库已作为 git submodule 接入：

- `third_party/cosmos-predict2.5`
- `third_party/cosmos-transfer2.5`
- `third_party/cosmos-reason2`
- `third_party/cosmos-rl`
- `third_party/lingbot-vla`
- `third_party/lingbot-va`
- `third_party/dreamzero`
- `third_party/Motus`

## 初始化方式

用下面的命令初始化或刷新全部子模块：

```bash
git submodule update --init --recursive third_party
```

如果只想初始化单个仓库：

```bash
git submodule update --init --recursive third_party/cosmos-predict2.5
```

## 为什么放在仓库根目录

当前仓库已经有 `verl/third_party/`，它主要用于 Python 导入兼容层和运行时补丁。新的 `third_party/` 故意与之分开：

- `verl/third_party/` 继续承担包内兼容和运行时 glue 的职责。
- `third_party/` 用来放上游外部研究仓库，尽量保持原始结构。

这样可以降低后续同步上游代码时的维护成本，也避免把 import shim 和完整外部项目混在一起。

## Cosmos 说明

当前 `verl.experimental.vla` 的接入路径默认使用 `simulator_type=cosmos`，并以 `env.train.cosmos.backend=mock` 作为 smoke test 后端。

代码里同时预留了未来接入 `third_party/cosmos-predict2.5` 官方路径的校验逻辑，但官方机器人 action-conditioned 推理目前仍偏向文件输入输出流程，且文档说明为单卡，因此这次补丁并没有把它声明为完整的生产级逐步在线推理实现。

## 推荐安装顺序

1. 初始化 `third_party/` 子模块。
2. 运行 `python -m verl.experimental.vla.prepare_cosmos_dataset` 生成最小数据集。
3. 用 `verl/experimental/vla/run_pi05_cosmos_sac.sh` 走通单机 smoke 路径。
4. 再用 `verl/experimental/vla/run_pi05_cosmos_sac_disagg.sh` 扩展到 train/env 分离节点。

## 当前集群里的 `cosmos-rl` smoke 路径

在当前 GH200 ARM64 环境里，最可复现的最小 `cosmos-rl` 路径是：

```bash
bash scripts/run_cosmos_rl_smoke_apptainer.sh
```

这个脚本会：

- 使用 `~/code/verl_docker` 里的 ARM64 apptainer 镜像
- 在容器里创建一个临时 `--system-site-packages` venv
- 只补齐最小 `cosmos-rl` SFT 路径缺失的 Python 包
- 为 `redis-server` 和 `torchrun` 提供兼容包装
- 使用 `third_party/cosmos-rl/tests/data_fixtures/sharegpt52k_small` 这个本地 fixture 数据集，运行一个 `1 GPU`、`1 step` 的 SFT smoke 示例

## `cosmos-rl` 是否能直接跑

在当前工作区环境里，`cosmos-rl` **不能** 在裸机宿主上直接运行。

主要原因包括：

- 宿主 Python 环境没有 GPU 版 `torch` 等关键运行时依赖
- ARM64 基础镜像已经比较接近，但依然缺少若干 `cosmos-rl` 所需的 Python 包
- controller 会直接调用 `redis-server`
- 这里采用的 `redislite` Redis 二进制不接受生成配置里的 `tls-port 0`，因此需要 wrapper 去掉这一行以保持兼容

因此更准确的实际结论是：

- 裸机宿主上不能直接跑
- 在 apptainer 镜像里加一个很薄的兼容层后可以跑

