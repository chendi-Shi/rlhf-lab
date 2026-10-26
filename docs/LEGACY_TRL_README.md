# RLHF Lab：小显存文本摘要对齐实验

**状态：项目代码、3,000 条真实偏好原始数据及试跑数据划分已准备，完整性检查已通过；尚未完成训练环境安装、GPU 试跑或效果验证。** 当前没有训练所得权重、吞吐量、显存峰值或胜率结果。项目以真实更新参数的 **SFT → 人类偏好奖励模型 → PPO** 为目标，供简历项目与实验复现使用。

基础模型选用约 124M 参数的 `openai-community/gpt2`，任务为英文摘要。GPT-2 的语言能力和小样本预算会限制最终效果；本项目应展示训练、数据治理和可靠评估能力。模型说明见 [GPT-2 model card](https://huggingface.co/openai-community/gpt2)。

## 准备内容

- 三个训练阶段分别保存模型，PPO 使用前两阶段的产物。
- `doctor` 检查资源，训练命令默认只预览计划，增加 `--execute` 后才执行。
- 公开人类摘要偏好数据准备、按提示词隔离数据、独立留出集和数据清单。
- 生成对照输出、匿名 A/B 评审材料，以及从人工判断计算胜率的工具。
- 小规模 `pilot` 配置与后续实验设计；正式规模应在试跑后调整。

训练后需要补充：实测环境、失败与成功运行日志、三阶段权重、独立测试结果和盲评记录。**当前不能把“效果提升”“已完成完整 RLHF 训练”写入简历。**

## 当前资源安排

最近一次对话中的检查显示 GPU 已满载、显存接近用满、可用系统内存约 2.1GB；这些是当时的快照。准备阶段通过轻量接口下载了约 5.15MB 的真实偏好数据；没有启动训练、占用 GPU、下载模型、安装训练依赖或停止其他进程。

后续先运行 `doctor` 获取实时状态。默认训练门槛为可用显存至少约 6GB、可用系统内存至少约 8GB、GPU 利用率不高于 20%，并检查 NVIDIA 驱动版本。这些门槛只用于减少资源冲突，**通过检查不等于保证 8GB 显卡不会 OOM**。PPO 同时涉及策略、参考、奖励和价值网络，需先测峰值显存。

## 现在就能做的轻量检查

以下命令使用本机已有嵌入式 Python，仅运行准备阶段测试、数据校验与计划预览，不安装训练依赖、不加载模型。`local_cli.py` 为隔离解释器补充本项目的导入路径；训练安装仍须使用后文的完整 CPython。

```powershell
& 'C:\Users\Intern1\Documents\ChatGPT\chenxi\.tools\python\python.exe' 'C:\Users\Intern1\Documents\ChatGPT\chenxi\rlhf-lab\scripts\check.py'
& 'C:\Users\Intern1\Documents\ChatGPT\chenxi\.tools\python\python.exe' 'C:\Users\Intern1\Documents\ChatGPT\chenxi\rlhf-lab\scripts\local_cli.py' validate-data --directory data/processed/pilot
& 'C:\Users\Intern1\Documents\ChatGPT\chenxi\.tools\python\python.exe' 'C:\Users\Intern1\Documents\ChatGPT\chenxi\rlhf-lab\scripts\local_cli.py' plan --config configs/pilot.json
```

## 后续安装环境

以下命令均需在项目目录中手动执行。环境脚本会联网下载依赖，请等资源与安装时机合适后再运行。

准备一个完整的 **64 位 CPython 3.11 或 3.12**，提供真实的 `python.exe` 绝对路径；不要使用现有 `.tools` 中的嵌入式解释器替代。脚本在项目自己的 `.venv` 中安装，不要求激活环境，不更改全局 Python、驱动或 PowerShell 执行策略。

```powershell
# 请替换为电脑上实际存在的完整 CPython 路径
.\scripts\setup.ps1 -PythonExe 'C:\Path\To\Python311\python.exe' -WhatIf
.\scripts\setup.ps1 -PythonExe 'C:\Path\To\Python311\python.exe'

.\.venv\Scripts\python.exe -m rlhf_lab doctor
.\.venv\Scripts\python.exe -m rlhf_lab plan --config configs/pilot.json
```

候选环境使用 PyTorch 2.8.0 的 CUDA 12.6 wheel，另固定 TRL 0.25.1、Transformers 4.56.2、Datasets 4.3.0、Accelerate 1.10.1 与 NumPy 2.2.6，详见 `requirements-train.txt`。这些是待实机验证的顶层版本约束，**不是已验证的完整依赖锁文件**。安装后保存 `pip freeze`、解释器版本与驱动信息作为运行记录。

[PyTorch 官方历史版本页](https://pytorch.org/get-started/previous-versions/)列出了对应 CUDA wheel。选用固定的 [TRL 0.25.1 PPO API](https://huggingface.co/docs/trl/v0.25.1/en/ppo_trainer)，更新 TRL 前需要重新检查接口。先前发现的旧驱动需在未来单独处理；本项目不会自动升级驱动。

## 数据准备

默认数据为 [CarperAI/openai_summarize_comparisons](https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons)，包含 `prompt`、`chosen` 和 `rejected`。偏好来源于公开的人类反馈数据，不能声称由本人标注。SFT 使用训练分区的优选摘要作为监督目标，奖励模型学习同一提示词下的摘要偏好，PPO 只使用训练分区的提示词。

**试跑数据已经就绪，无需再次下载或准备。** `data/raw/pilot_rows.jsonl` 保存官方 **train 前 3,000 条**比较，约 5.15MB；配套 `.provenance.json` 保存来源、版本核验、逐页响应哈希和逐行哈希。API URL 本身不锁定 revision，本次读取前后仓库 SHA 与各页 `x-revision` 均匹配预期版本，按已捕获的 API 快照记录，不把它描述为不可变的版本 URL。

`data/processed/pilot` 已清洗并按规范化原文分组创建本地划分，校验通过。**这里的 valid/test 都来自官方 train 前缀中的本地留出，不是官方验证集/测试集，也不是全数据集均匀抽样。**

| 数据 | train | valid | test |
|---|---:|---:|---:|
| 偏好比较对 | 1,024 | 128 | 128 |
| 不同原文 | 268 | 34 | 33 |
| SFT 监督样本 | 686 | 88 | — |
| PPO 提示词 | 268 | — | — |
| 最终摘要评估提示词 | — | — | 33 |

```powershell
# 已有数据只需检查，不会重新下载
.\.venv\Scripts\python.exe -m rlhf_lab validate-data --directory data/processed/pilot

# 后续扩大数据时，使用另一个配置与数据目录；此命令显式联网下载
.\.venv\Scripts\python.exe -m rlhf_lab prepare-data --config configs/experiment.json
```

`configs/experiment.json` 请求最多 8,192 条训练偏好，另请求 512 条验证、512 条测试偏好；该 Hub 路径使用官方 train/valid1/test，实际清洗后规模以清单为准。流式读取仍会下载数据块。现有输出目录受保护，重复 `prepare-data` 不会覆盖已有数据。

如需从当前原始文件重建试跑数据，先复制 `configs/pilot.json` 为新的配置，例如 `configs/pilot-rebuild.json`，把 `data_dir`、`output_dir` 分别改为全新的 `data/processed/pilot-rebuild`、`runs/pilot-rebuild`，再运行：

```powershell
.\.venv\Scripts\python.exe -m rlhf_lab prepare-data --config configs/pilot-rebuild.json --input data/raw/pilot_rows.jsonl
```

当前原始文件的来源侧文件已保留。换用其他本地 JSONL 时，必须同时提供 `<输入文件名>.provenance.json`，记录真实来源 `origin`、标注方式 `annotation_method`、`human_preferences: true` 与 `synthetic: false`。这些标记必须符合数据事实，不能把合成示例改个标记充当人工偏好。缺少来源声明的本地文件默认未验证，训练入口会阻止将其用于本项目的真实 RLHF 训练。

相同规范化提示词不得跨训练和评估分区；不要把当前 test 用于调参。将来更换数据方案时需重新检查与旧实验的原文重叠，不能沿用已看过或参与训练的样本来声称“全新独立测试”。

各阶段共用摘要边界：原文后追加 `\n\nTL;DR:`，监督摘要与偏好摘要开头已有的 `TL;DR:` 会去除。pilot 最多保留 192 个提示词 token 和 64 个回答 token（监督回答预算含 EOS），长原文会丢失较早上下文，回答也可能被截断。SFT、奖励模型、PPO 和评估使用同一格式；应在结果中披露截断限制，并检查关键信息是否被截掉。

## 分阶段真实训练

```powershell
# 仅显示计划，不加载训练模型
.\.venv\Scripts\python.exe -m rlhf_lab train --stage sft --config configs/pilot.json

# GPU 空闲、资源检查通过、环境与数据准备完成后，依次手动执行
.\.venv\Scripts\python.exe -m rlhf_lab train --stage sft --config configs/pilot.json --execute
.\.venv\Scripts\python.exe -m rlhf_lab train --stage reward --config configs/pilot.json --execute
.\.venv\Scripts\python.exe -m rlhf_lab train --stage ppo --config configs/pilot.json --execute
```

首次训练会下载基础模型。`pilot` 设置为 100 步 SFT、100 步奖励模型训练、128 个 PPO episode，用于验证三阶段能正常连接与保存模型，不能用于承诺效果。每个阶段结束后先检查日志和产物，再进入下一阶段。不要在另一个训练任务仍占用 GPU 时执行下一条命令，也不要一次粘贴全部命令启动。

资源错误时先缩短输入/生成长度、减少单卡批量，结合梯度累积重新测量；调整配置时保存新文件和新的输出目录。尚未验证中断恢复，不要把重复执行命令视为自动续训。

## 独立评估与人工盲评

训练完成后在相同的留出提示词、相同的生成设置上比较 Base、SFT 和 PPO。`generate` 默认使用 `--split valid`，调参期间使用验证集；确定最终配置后再执行以下 `--split test` 命令。下面建立 SFT/PPO 最终对照；Base 也应作为第三组按同样流程生成。

当前 pilot 有 33 个不同 test 原文，默认 `eval_limit` 只生成其中 32 条；这是流程验证规模，不能代替正式效果实验。扩大测试后再报告足够样本的盲评结果。奖励模型可单独评估，`evaluate-reward` 默认也用 valid，增加 `--execute` 后才会加载模型；下例在最终 test 上评估：

```powershell
.\.venv\Scripts\python.exe -m rlhf_lab evaluate-reward --config configs/pilot.json --model runs/pilot/reward --split test --output runs/eval/reward-test.json --execute
```

```powershell
.\.venv\Scripts\python.exe -m rlhf_lab generate --config configs/pilot.json --model runs/pilot/sft --label sft --split test --output runs/eval/sft.jsonl --execute
.\.venv\Scripts\python.exe -m rlhf_lab generate --config configs/pilot.json --model runs/pilot/ppo --label ppo --split test --output runs/eval/ppo.jsonl --execute
.\.venv\Scripts\python.exe -m rlhf_lab blind --left runs/eval/ppo.jsonl --right runs/eval/sft.jsonl --output-dir runs/blind
```

将生成的 `ballot.csv` 交给评审者，单独保管 `mapping.json`。先按提示词判断摘要是否忠实、完整、简洁，再在 `judgment` 列填写 `A`、`B` 或 `tie`；未判断的保留空白。保留原始列名与文本，评审理由另存备注文件。完成判断后运行评分工具。

```powershell
.\.venv\Scripts\python.exe -m rlhf_lab score --judgments runs/blind/ballot.csv --mapping runs/blind/mapping.json --output runs/blind/report.json
```

评分结果以 `--left` 模型为视角；上述例子报告 PPO 相对 SFT 的表现。必须同时报告样本数、平局、胜率口径和不确定性。奖励模型测试集上的偏好准确率反映其拟合偏好的能力；**训练奖励上升不能单独证明摘要质量提升**。独立人工盲评和失败案例是主要补充证据。详细规则见 [实验计划](docs/experiment-plan.md)。

## 预计投入与简历表述

在训练资源可用、已有 Python/PyTorch 基础且每天投入 4–6 小时的前提下：小数据三阶段跑通约 2–4 天；具备训练记录和独立评估的初版约 1–2 周；补全多组对照与失败分析约 2–4 周。环境适配和 GPU 排队时间另计。具体训练小时数须在本机完成首轮试跑后估算。

完成后可按事实填入：“基于 GPT-2 124M 完成摘要 SFT、人类偏好奖励建模及 PPO，对训练显存、KL 约束和盲评表现进行对照实验；在 **[实际样本数]** 条独立样本上取得 **[实测结果与口径]**。”未产生结果前保留占位，不写虚构百分比。

项目参考与出处见 [sources.md](docs/sources.md)。这里的工作重点是训练流程集成、资源约束、数据隔离与实验评估；PPO 核心算法由 TRL 提供。
