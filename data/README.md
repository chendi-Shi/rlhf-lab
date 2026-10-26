# 训练数据与来源

这里目前只有说明，没有下载数据，也没有生成用来冒充人工反馈的样本。

默认数据集是 [CarperAI/openai_summarize_comparisons](https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons)，来自 OpenAI 的[摘要人类反馈项目](https://github.com/openai/summarize-from-feedback)。每条记录包含文章 `prompt`、更受人工偏好的摘要 `chosen`、另一个摘要 `rejected`。候选摘要可以由模型生成；这里的“人类反馈”指比较偏好来自人工判断，不代表所有摘要都由人写作。

已核对的数据集版本：`a320fae2cd84f04610a1b57ac83c2a2f674b8bcd`。数据集仓库元数据提供 `train`（92,534）、`valid1`（33,082）、`valid2`（50,715）、`test`（83,629）四个划分。代码使用 train → train、valid1 → valid、test → test，暂不使用 valid2。

Hub 下载是显式调用 `prepare_hub()` 的网络操作。它按流读取每个来源划分的有限前缀，再按种子与文章哈希选取样本；每个前缀上限是 `max(2048, 32 × 该划分样本上限)`。这是控制资源的小样本方案，**不等于从完整数据集均匀随机抽样**。后续扩大实验时，应记录实际使用的前缀范围及划分规模。流式读取仍可能需要下载 Parquet 数据块，不保证网络字节数等于最后导出的 JSONL 大小。

同一文章的多个比较使用统一的归一化身份：Unicode NFKC、折叠空白、忽略大小写。处理会移除空字段、相同答案、重复偏好对和互相矛盾的 A>B/B>A 对。正式导出的 train、valid、test 文章互不重合；若来源划分重叠，保留 test，其次 valid，最后 train。生成文件保留正文大小写和内部换行。

| 文件 | 用途 |
| --- | --- |
| `sft_train.jsonl` / `sft_valid.jsonl` | 对 chosen 摘要做监督微调；不是独立人工示范语料 |
| `preferences_train.jsonl` / `preferences_valid.jsonl` / `preferences_test.jsonl` | 奖励模型训练、选择与最终成对评估 |
| `ppo_train.jsonl` | 仅来自训练划分的去重文章，用于 PPO 生成 |
| `eval_prompts.jsonl` | 仅来自测试划分的文章与偏好参考摘要 |
| `manifest.json` | 来源、版本、清洗计数、实际样本量、文件 SHA-256、划分方法 |

`chosen` 是比较标签，不是事实准确性保证。SFT 与奖励模型复用 train 是这个低资源起点的明确设计；最终报告中需要披露，不可将训练奖励提高直接写成人类偏好改善。

本地输入支持 UTF-8 JSONL，每行至少具有 `prompt`、`chosen`、`rejected` 三个非空字符串字段。`prepare(input_path, output_dir, ...)` 按文章分组，用确定性哈希顺序分成约 80/10/10 后再限制样本量。它不读取来源官方划分，因此不能声称保留了官方 train/test。

本地文件默认来源未验证，不能自动认定是真实人类反馈。若确实有人工偏好来源，在输入文件旁放置 `<输入文件名>.provenance.json`，记录 `human_preferences: true`、`synthetic: false`、`origin`（来源链接）、`annotation_method`（标注过程）。这些是用户声明，仍需自行核验真实性。代码测试里的合成数据仅存在于临时目录，不应作为训练成果。

`validate_dataset()` 会检查文件哈希、字段、条数、重复 ID、跨划分文章泄漏和派生文件一致性。数据集需先检查许可证、来源说明和公开演示中是否包含个人信息，再决定哪些原文例子适合展示；训练检查点与指标只有实际训练评估后才会产生。
