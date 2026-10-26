# AlignLab · Qwen3-1.7B 偏好对齐实验

AlignLab 研究单张 8 GiB GPU 上的语言模型偏好对齐：以 Qwen3-1.7B 为基座，使用 NF4 量化、LoRA、共享 SFT 与奖励模型，比较随机种子 42、43、44 的 DPO 和 PPO。仓库同时保留数据审计、模型接受、资源约束和失败追溯的实现。

**截至 2026-10-26：主训练已完成；独立质量评测尚未完成。** 最新 V4 尝试仅完成 base、SFT 两个模型的容量检查（2/9），随后在奖励模型最长输入上发生注意力显存不足。正式评测生成、评分和主结果分析均未开始。不能据此声称模型质量提升、PPO 优于 DPO，或项目评测已经成功。详细历史与证据指纹见 [评测状态](docs/EVALUATION_STATUS.md)。

## 实验范围

- 基座：Qwen3-1.7B；模型参数与数据来源通过清单和 SHA-256 固定。
- 方法：共享 SFT、共享 RM；DPO 与 PPO 分别运行 seeds 42/43/44。NF4 基座与 LoRA 参数分开管理，执行中检查参数只读及产物身份。
- 训练：SFT 完成 250 次更新、4,000 条样本；RM 完成 822 次更新、13,139 对；每个 DPO seed 完成 822 次更新、13,139 对；每个 PPO seed 完成 2,048 次更新、1,024 个真实 prompt。六组 DPO/PPO 主训练及最终接受均已通过。
- 独立评测对象：base、SFT、RM、DPO42/PPO42、DPO43/PPO43、DPO44/PPO44，共九个模型。
- 固定评测：八个策略模型各 541 个 IFEval prompt、834 条指令，共 4,328 次生成；RM 使用 2,017 对 HelpSteer3 留出数据和 1,327 条 RM-Bench 记录。
- 生成约束：greedy、thinking=false、max_new_tokens=2048、不截断输入；官方评分 seed=0、bootstrap=2000，保留完整分母。

没有新增人工标注；人工标签数为 0，人工胜率为 `null`。RM 分数是诊断量，不能直接当作回答质量提升。

## 目录

| 路径 | 内容 |
| --- | --- |
| `alignment_lab/` | 当前 AlignLab 的训练、接受、评测、资源守护与版本化恢复实现 |
| `scripts/` | 审计、准备、CPU 验证及显式执行入口 |
| `tests_v2/` | AlignLab 的数学、身份、资源、评测和恢复回归测试 |
| `rlhf_lab/`、`tests/` | 历史 GPT-2 / 小模型摘要 RLHF 实现与测试 |
| `tests_throughput_v1/`、`tests_throughput_v2/` | 历史吞吐与恢复测试 |
| `configs/` | 发布的通用配置；属于 legacy 路线，不是当前冻结的 Qwen3 启动计划 |
| `docs/` | 当前状态、发布范围与历史设计文档 |
| `data/README.md` | 数据使用说明；下载数据不随源码发布 |

历史设计入口为 [AlignLab V2 设计](docs/ALIGNLAB_V2.md)。它保留当时的方案和修订背景，不代表当前运行状态；最新结果口径以本 README 和 [评测状态](docs/EVALUATION_STATUS.md) 为准。

## 环境与入口

`pyproject.toml` 声明 Python `>=3.11,<3.13`，注册的命令只有 `rlhf-lab = rlhf_lab.__main__:main`；项目元数据没有自动安装训练依赖。查看命令帮助不会下载模型或启动训练：

```powershell
python -m rlhf_lab --help
```

当前 Qwen3 执行链检查的关键版本为 `torch==2.7.1+cu118`、`transformers==5.8.1`、`peft==0.18.0`、`bitsandbytes==0.50.2`、`accelerate==1.12.0`。这些是已有执行环境的身份约束，不是对任意操作系统和 GPU 的安装兼容性承诺。

依赖文件的用途不同：`requirements-local.txt` 对应本地引擎依赖，`requirements-dev.txt` 增加 pytest，`requirements-v2-audit.txt` 用于 CPU 数据/分词审计；`requirements-train.txt` 是历史 TRL 路线的候选依赖，不能用它替代当前 Qwen3 环境。模型、分词器、数据和运行环境均需另行准备，仓库不会因克隆或查看帮助而自动获取这些资源。

`rlhf-lab` 的 `local-*` 命令及发布的通用配置面向历史 GPT-2 / 小模型路线；当前 Qwen3 的执行入口位于版本化 `scripts/` 文件中，需要冻结计划、输入证据、当前 CPU 检查和真实 CUDA 容量验收。发布源码不需要运行任何 `start`、`supervise` 或训练脚本。

## CPU 检查

在依赖已准备的项目环境中，可显式运行不加载预训练权重的资源回归：

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m unittest tests_v2.test_evaluation_resources_v4 -v
```

完整 V4 CPU 证明入口是 `scripts/verify_alignlab_evaluation_v4.py`，它要求指定的隔离解释器、隐藏 CUDA 和全新的报告目录。该脚本在本地对应源码上曾通过 77 项检查；CPU 通过不代表 CUDA 容量通过或质量评测完成。发布整理期间没有重跑测试，也没有启动模型或评测。

## 发布范围

这是源码仓库，首次发布的源码和测试约 4.7 MB。模型权重、下载数据、运行产物、原始日志、报告和机器专用启动计划留在原执行环境，未包含在 GitHub 提交中。保留的证据文件名与指纹用于追溯，并不意味着原始证据已公开。详见 [发布说明](docs/PUBLISHING.md)。
