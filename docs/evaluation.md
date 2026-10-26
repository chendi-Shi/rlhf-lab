# 真实生成与人工盲评

本项目准备阶段没有训练结果、模型权重或人工胜率。测试中的短文本只是验证程序正确性的合成数据，不能用作简历中的模型效果。

## 实验设计

1. 固定训练、验证和评估集合；同一来源文章不得跨集合出现。
2. 完成 SFT 后保存权重，再训练奖励模型与 PPO。调参用验证集，最终方案确定后再使用保留测试集。对 Base、SFT 和 PPO 使用相同评估文章、输入长度、生成长度和解码设置。
3. 主对照是 **PPO 与 SFT**。Base 与 SFT 用来说明监督微调带来的变化，不能替代 PPO 对照。
4. 在独立的人类偏好验证集合上报告奖励模型成对准确率；在独立生成评估集合上进行人工盲评。两种评估回答不同的问题。
5. 先用 32 条完成流程和发现明显错误，再扩大到约 100–200 条人工评估。样本量只是实验安排，不是效果保证；报告实际评估数量和不确定性。

奖励模型分数上升只说明模型更符合该奖励模型的偏好，可能包含奖励投机。不能把奖励分数当作人工胜率或事实正确率。英文 GPT-2 与英文摘要数据的实验也不能外推为中文能力。

## 生成文件约定

真实生成函数是 `rlhf_lab.generate.generate(config, model_path, label, output_path, limit=None)`。它先检查资源，再延迟加载 PyTorch 和 Transformers，加载基础模型或已训练的本地模型权重进行自回归生成。准备阶段不调用此函数。

CLI 的 `generate --split valid` 是默认选项，使用 `preferences_valid.jsonl`，将 `chosen` 保存为参考摘要；`--split test` 使用 `eval_prompts.jsonl`，留到最终评估时再运行。Python API 对应 `config["eval_split"]`，默认也是 `valid`。两个分区均按规范化文章去重，保留准备文件中首次出现的 ID 与参考摘要，再应用数量上限，避免同一原文的多组偏好比较被重复计入盲评。

每行 JSON 包含：

| 字段 | 含义 |
| --- | --- |
| `id` | 非空字符串，标识相同评估文章 |
| `prompt` | 原始完整文章，各模型必须一致；不是模型实际看到的完整 token 序列 |
| `response` | 实际生成的续写，不包括输入 |
| `model_label` | 例如 `base`、`sft`、`ppo` |
| `model_path` | 基础模型标识或本地权重路径 |
| `seed` | 本条生成使用的随机种子 |
| `reference` | 可选的人类参考摘要 |

生成采用贪心解码，每次一条，限制新生成 token 数。输入与训练共用函数：为原始文章追加 `\n\nTL;DR:`，编码后只保留最后 `max_prompt_tokens` 个 token，保留摘要指令。默认最多 192 个输入 token 与 64 个新生成 token；SFT/RM 的目标摘要上限包含 EOS。训练目标还会移除摘要开头的一次 `TL;DR:` 并补一个前导空格，使 GPT-2 的任务边界一致。

**评审 CSV 展示完整原文，而模型可能只看到末尾窗口。** 这里衡量的是该截断方案下相对于完整文章的摘要表现；模型遗漏已被截去的信息可能反映输入限制。必须在结果中披露，不能把它解释为完整上下文的摘要能力。所有对照模型使用相同截断预算。种子用于记录和控制初始化，跨硬件或软件版本仍可能有数值差异。

输出文件不覆盖已有文件。生成意外中断时可能留下部分 JSONL 与状态为 `running` 的清单，应保留用于排错并用新文件名重跑；未完成清单会阻止导出盲评。只有 ID 完整匹配的两份输出才能导出对照盲评。

生成前校验准备数据的文件哈希和分区隔离。本地权重必须具有状态为 `completed` 的 SFT 或 PPO 运行清单，且基础模型、版本、数据清单、token 预算与种子均匹配。生成完成后额外保存 `<输出文件>.manifest.json`，记录原模型版本、检查点清单哈希、分词器文件哈希、评估输入哈希、配置、依赖、资源检查与实际耗时。基础模型的 PAD token 按固定种子添加；本地模型直接读取训练阶段保存的分词器。

## 导出盲评

Python API：

```python
from rlhf_lab.evaluate import export_blind_pairs, score_judgments, summarize_generations

export_blind_pairs(
    "runs/pilot/eval/ppo.jsonl",  # left：统计中的主模型
    "runs/pilot/eval/sft.jsonl",  # right：对照模型
    "runs/pilot/eval/ballot.csv",
    "runs/pilot/eval/mapping.json",
    seed=42,
)
```

导出前验证两份文件 ID 完全匹配、无重复、相同 ID 的 prompt 相同，且各文件内部模型身份一致。空回复允许进入评估，不会被悄悄过滤。随机种子控制逐条 A/B 排列以及展示顺序；调整输入 JSONL 行顺序不会改变结果。

本项目生成的 JSONL 需连同各自 `.manifest.json` 保留。存在任一侧清单时，两侧都必须有完成的清单，输出文件校验和、数据版本、分区、输入文件、生成方式、推理精度和 token 预算等必须匹配。外部生成文件若两侧都没有清单，仍支持导出，但实验负责人需另行核实并记录相同设置。

CLI 的 `blind --output-dir ...` 在指定目录生成 `ballot.csv` 和 `mapping.json`。`ballot.csv` 只含 `pair_id,prompt,response_a,response_b,judgment`，不含模型标签、检查点路径或模型身份映射。**只把此 CSV 交给评审者**；`mapping.json` 由实验负责人单独保存，不与评审者共享。该方案隐藏元数据身份，无法保证模型文风或内容完全不可辨认。

填写 `judgment`：

- `A`：A 更好。
- `B`：B 更好。
- `tie`：质量接近，无法可靠判断。
- 空白：尚未评估；不计作平局或胜利。

只修改 `judgment` 列，不删除行或修改输入、输出。程序通过哈希检测文本被改动的情况，也拒绝未知、重复或丢失的 pair_id。

评审者先读文章，再比较摘要的忠实性、关键内容覆盖、简洁性和可读性；有明显杜撰事实的摘要应优先判为较差。两份都差时仍可判相对更好的一份，无法分辨则平局。不要仅凭更长、更流畅或迎合措辞判胜。可以另行记录错误类型和理由，不将其写入固定 CSV 结构。条件允许时请两名评审者分别填写独立副本，单独分析分歧，再以事先约定的方式形成最终判断。

## 汇总与解释

```python
report = score_judgments(
    "runs/pilot/eval/ballot.csv",
    "runs/pilot/eval/mapping.json",
    "runs/pilot/eval/human_report.json",
)
proxies = summarize_generations(
    "runs/pilot/eval/ppo.jsonl",
    "runs/pilot/eval/text_diagnostics.json",
)
```

所有胜负以导出时的 **left 模型** 为视角，并在报告中保留两侧身份。报告包含实际胜、平、负、尚未评估数量，且明确分母：

- `decisive_win_rate = wins / (wins + losses)`，排除平局；附该二项比例的 95% Wilson 区间。
- `tie_adjusted_score = (wins + 0.5 * ties) / judged`，单独展示，不套用上述二项区间。
- 没有明确胜负时，decisive win rate 和区间为 `null`；没有人工判断时，各胜率为 `null`。不会生成占位胜率或自动把奖励分数变成人工票。

Wilson 区间假设独立样本，仅反映该评估样本的不确定性。相近文章、同一评审者的系统偏差或挑选有利样本会影响解释。扩展训练之前固定评估集合，并同时报告平局比例、覆盖数量和失败例子。单次小样本优势不足以证明通用能力提升。

文本诊断仅报告回复长度、空回复数量及重复三元组比例：`(三元组总数 - 不同三元组数) / 三元组总数`。分词采用 Unicode 正则，属于简单代理指标，英文与中文的分词行为不同；它不是模型 tokenizer 的 token 计数，也不是质量、事实准确性或人工偏好的评分。

简历中的数值必须来自保留的真实运行记录与上述人工报告。建议同时保留配置、数据版本、代码版本、检查点、资源测量、训练日志和匿名评估文件，并解释负面结果与限制。
