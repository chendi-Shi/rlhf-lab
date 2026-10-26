# 来源与实现边界

核查日期：2026-10-16。上游效果数字不属于本项目结果。当前 `local-*` 为独立 PyTorch 实现；旧 `train` 路径仍使用 TRL。

| 来源 | 在本项目中的用途 |
|---|---|
| [ttumiel/minRLHF](https://github.com/ttumiel/minRLHF)（MIT） | 124M GPT-2 摘要 RLHF 的概念与项目结构参考。本项目不是该仓库原样复现。 |
| [Hugging Face TRL](https://github.com/huggingface/trl)（Apache-2.0） | `v0.25.1` PPO/DPO 源码用于算法核对；当前本机训练器不调用 TRL，旧训练器仍依赖它。 |
| [DPO 论文](https://arxiv.org/abs/2305.18290) | 带参考策略的偏好对数比分类目标。 |
| [PPO 论文](https://arxiv.org/abs/1707.06347) | on-policy 采样、概率比裁剪与多轮 minibatch 更新。 |
| [TRL 0.25.1 PPO 文档](https://huggingface.co/docs/trl/v0.25.1/en/ppo_trainer) | PPOTrainer 参数、模型角色、训练指标和批量设置参考。 |
| [GPT-2 模型卡](https://huggingface.co/openai-community/gpt2)（模型卡标注 MIT） | 124M 英文预训练基础模型；本项目不进行从零预训练。 |
| [CarperAI/openai_summarize_comparisons](https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons) | `prompt/chosen/rejected` 摘要偏好数据。公开的人类比较数据，不是本项目自行标注的数据。 |
| [OpenAI summarize-from-feedback](https://github.com/openai/summarize-from-feedback) | 原始摘要人类反馈研究、数据背景及方法参考。 |
| [PyTorch 历史安装版本](https://pytorch.org/get-started/previous-versions/) | PyTorch 2.8.0 与 CUDA 12.6 wheel 的官方安装命令来源。 |

外部代码、模型和数据分别遵守各自的许可及使用条件；数据页面缺少完整说明时，继续核查原始发布信息，不能因为某个代码仓库采用 MIT 就推定全部数据同样采用 MIT。

试跑数据实际通过 Hugging Face 官方 datasets-server rows API 获取官方 train 的前 3,000 条；精确分页 URL、响应字节哈希、逐行哈希和仓库元数据核验记录在 `data/raw/pilot_rows.jsonl.provenance.json`。本次下载前后仓库 SHA 与全部响应的 `x-revision` 均匹配 `a320fae2cd84f04610a1b57ac83c2a2f674b8bcd`。该接口 URL 没有锁定 revision，因此将其记录为已捕获并核验的 API 快照，不声称为不可变下载地址。

原始 JSONL 的 SHA-256：`6bf99066a5a8d30a043e5cdd4345ffc88862104d0ae707cde95ae4acc98c4d95`。本地试跑 valid/test 是从这一官方 train 前缀按原文分组留出的样本，不是官方 held-out 分区。

本地训练器、目标函数、数据处理、恢复和评估属于项目实现；DPO/PPO 方法本身不是项目原创算法。增加第三方源码时保留相应版权与许可通知。
