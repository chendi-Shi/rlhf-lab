# 本机人类偏好数据与 GPT-2

本页记录 2026-10-16 的只读数据审查和固定模型下载。它不是训练完成报告，也不把数据校验当作模型能力验证。

## 数据身份与划分

现有 `data/raw/pilot_rows.jsonl` 来自 [CarperAI/openai_summarize_comparisons](https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons)，原研究为 [OpenAI summarize-from-feedback](https://github.com/openai/summarize-from-feedback)。它是已有人工摘要偏好的 chosen/rejected 重新打包，不是本项目新标注，也不是人工优质摘要演示集。

来源为官方 **train 前 3,000 行**的有序前缀，不是全库均匀抽样。抓取脚本在请求前后核对库 revision `a320fae2cd84f04610a1b57ac83c2a2f674b8bcd`，并保留每页 x-revision、响应摘要及逐行哈希。rows API 的 URL 本身没有 revision 参数，故只能称为捕获的 API 快照，不能声称响应字节由不可变 revision URL 提供。

本次运行 `rlhf_lab.data.validate_dataset("data/processed/pilot")` 通过。原始文件 SHA-256 为 `6bf99066a5a8d30a043e5cdd4345ffc88862104d0ae707cde95ae4acc98c4d95`，与 manifest 和 provenance 声明相符；3,000 条 canonical row SHA-256 全部匹配。provenance 文件 SHA-256 为 `88362e1246e5c4a931f126313c2c1c9c5f92bd98ec1fe80316b8a2a8e8805529`，与 manifest 相符。本次没有重新下载原始 API 页，哈希证明的是当前本地证据一致性。

清洗 3,000 行时，没有空值或相同 chosen/rejected；移除 972 条重复，以及 172 条互相矛盾的偏好记录，剩余 1,856 条可用比较。按 NFKC、空白归并、casefold 的 prompt 标识分组，使用确定性的 SHA-256 顺序分区后再限制样本数量。

| 文件／划分 | train | valid | test |
|---|---:|---:|---:|
| 偏好比较对 | 1,024 | 128 | 128 |
| 独立归一化 prompt | 268 | 34 | 33 |
| SFT preferred 摘要 | 686 | 88 | 不生成 |
| PPO prompt | 268 | 不生成 | 不生成 |
| 最终生成评测 prompt | 不生成 | 不生成 | 33 |

这些是从官方 train 前缀内部新建的本地 train/valid/test，不是官方保留 split。验证器重新检查文件 SHA、schema、ID 唯一性、归一化 prompt 跨划分不重叠，以及 SFT/PPO/eval 派生文件与原偏好一致。尚未检测语义近重复或同一文章的非精确变体；128 个测试比较不能当作 128 个独立 prompt。统计区间宜按 prompt 聚类；小保留集不能支持宽泛的能力或上线结论。

## 真实 tokenizer 下的截断审查

使用下述真实 GPT-2 tokenizer，针对原 `configs/pilot.json` 的 prompt 192 / response 64 配置，只检查 train/valid 的 token 长度和结构，未进行质量评测或查看 test 质量。

| 项目 | train（1,024 对） | valid（128 对） |
|---|---:|---:|
| prompt 超过 192 token | 973 | 115 |
| chosen 正文超过 63 token | 35 | 4 |
| rejected 正文超过 63 token | 17 | 2 |
| 截断后 chosen/rejected token 完全相同 | 0 | 0 |

原 `encode_prompt` 在末尾添加 `\n\nTL;DR:`，从左边截掉较早文章内容，保留摘要任务边界。因此这次 pilot 大部分例子只能看到文章末段，结果不能表述为完整文章摘要表现。

原 `encode_response` 去掉已有 TL;DR 前缀，将正文截到最多 63 token，再追加 EOS（ID 50256）。这保证目标最多 64 token，但对于被截断的正文，EOS 是人为终止标记，并非原摘要自然结束。若新训练器调整该政策，需要记录新协议和新截断统计，不可把旧表直接当作新训练的统计。

官方 GPT-2 tokenizer 没有 pad token。若本机训练入口令 pad_token=eos_token，应根据真实序列长度或 attention mask 屏蔽 padding，保留真实目标 EOS 的监督；不能仅按 token ID 将所有 EOS 一并屏蔽。

新的 `local-smoke` / `local-gpu` 配置已选择 `completion_eos_policy="natural"`：正文超过预算时保留最多预算数量的 token，不给截断正文伪造 EOS；正文恰好占满预算时也没有额外 EOS 的位置。2026-10-16 用同一真实 tokenizer 重新读取 train/valid 得到下表（不是训练或质量评测）：

| 配置 / split | prompt / response 预算 | 比较对 | prompt 被截断 | chosen 正文被截断 | rejected 正文被截断 | chosen / rejected 保留 EOS |
|---|---|---:|---:|---:|---:|---:|
| local-smoke / train | 96 / 24 | 1,024 | 1,024 | 637 | 511 | 357 / 453 |
| local-smoke / valid | 96 / 24 | 128 | 128 | 88 | 76 | 36 / 50 |
| local-gpu / train | 192 / 64 | 1,024 | 973 | 26 | 13 | 989 / 1,007 |
| local-gpu / valid | 192 / 64 | 128 | 115 | 4 | 2 | 124 / 126 |

两套配置在 train/valid 中均无截断后 chosen/rejected 完全相同的比较对。表中正文被截断的严格条件是 `正文长度 > response 预算`；正文恰等于预算时不计正文截断，但也不包含 EOS。因此 EOS 数量不必等于总数减去截断数。CPU smoke 的每个 prompt 都截断、且很多目标摘要截断，只能用于实际更新和执行流程验证。GPU pilot 也主要观察文章末段，仍不能支持完整文章摘要质量结论。

## 固定官方预训练模型

来源：[openai-community/gpt2 固定 revision](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e)。revision 为 `607a30d783dfa663caf39e06633721c8d4cfcd7e`。配置为 12 层、12 attention heads、768 hidden size、50,257 词表、1,024 最大位置。GPT-2 是较早的英文因果语言模型；用于本机可训练的 RLHF 对照，不称为当前先进模型。

已读取 [Hugging Face 固定 revision API](https://huggingface.co/api/models/openai-community/gpt2/revision/607a30d783dfa663caf39e06633721c8d4cfcd7e?blobs=true) 的官方文件身份：

- `model.safetensors`：548,105,171 bytes。
- 官方 LFS SHA-256：`248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`。
- 同 revision 下载 `config.json`、`generation_config.json`、`tokenizer_config.json`、`tokenizer.json`、`vocab.json`、`merges.txt`、上游 README。普通文件以官方 Git blob SHA-1 校验，另记录本地 SHA-256。

`scripts/download_local_model.py` 使用标准库和 4 MiB HTTP range 请求，默认 4 线程，可复用分块。最终权重必须同时通过文件长度与上述官方 LFS SHA-256 才发布为 `models/gpt2/model.safetensors`；仅有 `.download-parts` 或 `.part` 不算模型就绪。脚本不下载 `pytorch_model.bin`，不加载 pickle，不运行远端模型代码，不改安装环境，不调用 GPU。

```powershell
& '..\evidoc\.tools\python\python.exe' scripts\download_local_model.py --workers 4
& '..\evidoc\.tools\python\python.exe' scripts\download_local_model.py --verify-only
```

官方元数据保存在 `models/gpt2/upstream-files.json`；全部文件通过校验后生成 `models/gpt2/local-manifest.json`。这两个 JSON 与模型目录被 Git 忽略。训练应使用此本地目录、`local_files_only=True`、`trust_remote_code=False`、safetensors 权重。下载完成与实际训练／评测状态分别记录。

本机状态：2026-10-16 已完成全部 131 个分块，整合后的权重通过官方 SHA-256；随后离线 `--verify-only` 再次检查全部 8 个模型／tokenizer／说明文件通过。未用随机权重或 tiny 模型代替。下载与本页数据检查没有执行训练或 GPU 运算。
