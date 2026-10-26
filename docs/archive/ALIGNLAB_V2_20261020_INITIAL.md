# AlignLab 单卡大模型偏好对齐与奖励投机研究

2026-10-20 设计版。面向算法／机器学习工程师作品集，目标是在 RTX 3060 Ti 8GB 上实现现代指令模型的 SFT、奖励建模、DPO 与 PPO，并用独立自动评测回答：**有限计算预算下，两种偏好对齐方法的收益、成本和失效方式有什么差别？**

当前状态：方案已设计；新模型容量、训练吞吐和效果尚未验证。原 GPT-2 实验已取消后续调度，只保留历史记录。本文的规模和预算是待容量校准后冻结的设计目标，不是已完成成果，也不是可直接执行的 campaign 配置。用户要求不新增人工评测，训练顺序仍为新 RLHF 项目先于 VLA。

## 研究问题与项目价值

主问题是同一模型、SFT 起点、偏好数据和评测协议下，DPO 与 PPO 的计算成本—表现曲线。次问题是正则化变弱时，训练奖励、指令遵循和长度／重复行为是否出现分离。两个问题均允许负结果：没有证据时不宣称 PPO 优于 DPO，也不预设对齐一定改善已有指令模型。

项目贡献由三部分组成：8GB 上真正可运行的 PPO/DPO 训练系统；带数据隔离、三 seed 和成本记录的实验；可追溯到原始输出的奖励投机分析。标签噪声、跨领域重训练、额外算法和付费模型裁判不纳入首版，避免实验规模失控。

## 模型与数据

| 项目 | 设计选择 |
|---|---|
| 主模型 | `Qwen/Qwen3-1.7B`，固定模型和 tokenizer revision，统一 `enable_thinking=False` |
| 参数训练 | 冻结 NF4 基座，LoRA r=16，alpha=32；目标模块、精度和 dropout 在容量验收后冻结 |
| 兼容性备选 | `Qwen/Qwen2.5-1.5B-Instruct`；只有主模型确有兼容性问题才另建实验身份，不混入主模型的统计 |
| 原始偏好数据 | `nvidia/HelpSteer3` 的 preference 配置；38,459 条 train、2,017 条 validation |
| 数据范围 | 主数据覆盖 General、STEM、Code、Multilingual；分领域、语言和长度报告覆盖情况 |
| SFT 目标 | 最多 4,000 个去重 prompt 的高置信 chosen 回答；至少 2,000 个合格 prompt 后才进入正式设计冻结 |
| RM／DPO 数据目标 | 同一份最多 24,000 对分层抽样偏好；至少 8,000 个独立 prompt 簇、12,000 个有效非平局偏好对 |
| PPO 输入目标 | 从同一偏好训练集的全部合格 prompt 池采样；4,096 次 rollout 为起始工作量目标，实际独立 prompt 数、重复访问和 token 数另记 |

Qwen3-1.7B 是已有预训练和后训练的模型，因此本项目研究进一步偏好对齐，不声称从零构建通用聊天模型。[官方模型卡](https://huggingface.co/Qwen/Qwen3-1.7B)

HelpSteer3 的偏好由人工标注，候选回答由模型生成；总计约四万条比较不等于四万个独立 prompt。只使用 preference 配置，不混入 AI 生成的 principle 配置。SFT 数据应称“经人类偏好选择的模型回答”，不能称“人工撰写的标准答案”。保留 CC-BY-4.0 署名和数据来源。[官方数据卡](https://huggingface.co/datasets/nvidia/HelpSteer3)

### 划分和标签协议

1. 保留官方 validation 为本项目最终封存评测集；准确称为“官方 validation 用作项目最终留出集”，不称官方 test。从官方 train 按 prompt 簇分层划分 90% train、10% internal-dev，再从 internal-dev 预留诊断簇，余下部分称 tuning-dev。train、tuning-dev、diagnostic-holdout 三份清单互斥，固定划分 seed 和 manifest。
2. 完整上下文规范化哈希结合用户轮次／共享对话前缀和近重复聚类；跨分区冲突保留评测侧、隔离训练侧。近重复规则和阈值先在 train 确定。审计不等于证明预训练数据无污染，公开 benchmark 可能已被基座见过。
3. 按 `overall_preference` 的官方符号定义形成 chosen/rejected。零分平局单列，不强造正负标签；`individual_preference` 用于一致性分层，不作为模型输入。SFT 采用绝对偏好强度至少 2 的样本，每个 prompt 最多一条；强偏好不代表答案绝对正确。
4. 在 train 上审计真实 token 长度。正式目标为 prompt≤1,024、每个回答≤512 token，按完整 chat template 计数。超长训练对先过滤并记录原因，不截断答案后仍照搬原偏好标签。512+128 仅可用于初始算子探测，不能直接充当正式容量验收。
5. 数据不足最低规模时先报告领域／长度覆盖和显存瓶颈，调整后创建新版协议；不得把数百条 prompt 的结果改名为正式项目。各 seed 不重新划分数据，也不按文件前缀取样。
6. 最终留出集只在协议和 checkpoint 选择规则冻结后评分。评测可使用更长的推理上下文；不能因为某方法表现差而删样本。无法完整处理的样本采用事先规定的共同可评估集合，明确分母、排除原因和覆盖率，保留全量清单。

以上样本规模仍须 CPU 数据审计确认。达到数据入库目标与实际完成训练覆盖是两件事，报告必须同时列出两者。

## 8GB 上的训练架构

```text
Qwen3 量化冻结基座 ── SFT adapter ──┬── DPO policy adapter
                                  ├── 冻结 reference adapter
                                  └── PPO policy adapter ＋ value head
人类偏好 train ── RM adapter ＋ scalar head ── 冻结后为 PPO 评分

internal-dev：调参与固定成本检查点
封存偏好集 ＋ IFEval ＋ RM-Bench：最终自动评测
```

同一个冻结量化基座串行切换 adapter，避免四份完整模型同时常驻显存。SFT、reference、policy、RM 的参数和产物身份独立；RM 从固定 SFT adapter 副本训练，训练完复制冻结，PPO 不继承 DPO 权重。PPO actor 与 value head 共享 policy backbone。

DPO 使用 completion mask 和冻结 SFT reference 的逐回答 log-prob 缓存。缓存键覆盖模型／adapter、tokenizer、模板、token IDs、截断规则和源码哈希；改变任一项即失效。PPO 每轮依次完成当前策略采样、reference 与 RM 无梯度评分、固定 old log-prob／return／advantage、GAE 和 clipped PPO 更新。终止与长度上限截断的处理必须明确且一致。[DPO 双 adapter 参考](https://huggingface.co/docs/trl/v0.24.0/dpo_trainer#using-option-3---load-the-adapter-twice)、[PPO 参考实现](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/ppo_trainer.py)

训练 RM 的标量奖励加 KL 约束是 PPO 的优化信号；IFEval 和其他规则测试仅用于评测，不偷换成规则奖励训练。PPO rollout 必须来自当前 policy，不能用固定离线回答冒充 on-policy 采样。

### 必须解决的工程问题

- 有梯度的 forward 到 backward 完成期间禁止切 adapter；梯度检查点的重算必须使用同一 adapter。每次切换后核查 `requires_grad` 和 optimizer 参数集合，不跨 adapter 复用 KV cache。
- 保留训练模式以启用真实梯度检查点，并明确关闭需要关闭的 dropout；不能继续沿用旧训练器中 `eval()` 导致检查点无效的做法。冻结 reference/RM 的权重哈希在每个阶段前后保持一致。
- RM 取最后一个真实回答 token 的隐藏状态；prompt、padding 和非回答轮次不参与 completion loss，EOS／pad／截断边界均有数值测试。
- 在独立 `rlhf-lab/.runtime/alignlab-v2` 环境安装并锁定依赖。保持全局驱动和 EviDoc 等其他项目环境不变。普通 PyTorch／Transformers／PEFT 为主路径，TRL 固定版本用于公式和行为对照，不假设最新 TRL 的 PPO API 可直接使用。
- GPU 峰值同时记录 allocated、reserved 和设备实际占用；日志包含独立 prompt 数、生成／训练 token、吞吐、采样耗时、RM/reference 成本。4-bit 参数量估算不等于实际显存。
- 对大词表采用 completion token 的 LM-head 分块与经过数值验证的重计算，避免先产生全序列 FP32 logits 再切片。两条各 1,536-token 的完整输入与回答序列对应的全词表 FP32 logits 单份即可约占 1.87GB；必须核对分块与完整实现的 loss 和梯度，并让正式容量测试覆盖 DPO 两条最大长度回答。
- 如共享 RM adapter 存在无法排除的正确性问题，可设计独立 RM 串行卸载版本，但必须另行验证吞吐、RM 可靠性和实验身份。不得静默换成更小策略模型并报原设计完成。

Windows 的 bitsandbytes 有相应 CUDA 构建，但这台机器的驱动、量化算子、优化器和完整 PPO 周期仍须实测。[官方安装矩阵](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation)

## 主实验和消融

| 实验 | 数量 | 控制方式 |
|---|---:|---|
| 原始 Qwen3 指令模型 | 1 个基线 | 固定 non-thinking 模板，无本项目训练 |
| 公共 SFT | 1 次 | 所有对齐分支共享同一 checkpoint |
| 公共 RM | 1 次 | 所有 PPO 分支共享同一冻结 RM |
| DPO 主实验 | 3 次 | seeds 42／43／44，beta=0.1 |
| PPO 主实验 | 3 次 | seeds 42／43／44，初始 KL 系数=0.05 |
| DPO 单变量消融 | 2 次 | seed 42，beta=0.05／0.2 |
| PPO 单变量消融 | 2 次 | seed 42，KL 系数=0／0.1 |

主实验共有原始模型、SFT 和六个对齐模型；扩展另有四个短预算消融。三 seed 仅覆盖对齐阶段随机性，不能声称测量了整个 SFT／RM 管线的种子方差。beta 和 KL 系数数值没有相同强度的含义，另在固定 dev prompts 上估计实际 policy/reference KL。

候选优化设置：SFT lr=1e-4、RM lr=5e-5、DPO lr=5e-6、PPO lr=1e-6，microbatch=1，SFT／RM／DPO effective batch=16；PPO rollout batch=16、minibatch=1、2 epochs、clip=0.2、gamma=1、GAE lambda=0.95，生成目标上限256 token。它们是稳定性校准起点，正式运行前与 reward normalization、value clipping、采样和总更新数一起冻结。只用 train/tuning-dev 校准，不看最终留出集挑参数。

PPO 主实验采用 temperature=1、无 top-k/top-p 截断的 categorical sampling，old/new log-prob 和 reference log-prob 均使用同一 token 概率定义。任何 logits processor 都必须纳入分布定义，不能以经过截断的策略采样却用原始 softmax 计算 PPO ratio。tuning-dev 可用于校准，diagnostic-holdout 不参与参数、RM checkpoint 或停止阈值选择。

主实验记录约 1／2／4／8 个活跃运行小时处的完整检查点和真实耗时，dev 曲线使用预先固定的子集。消融在相同约 2 小时成本处与主实验对应检查点比较；不能把短消融与长主实验终点直接解释成算法差异。任何未实际达到的时间点留空。

一次数据遍历提前结束时记录 `stop_reason=data_exhausted`，不重复数据凑满时间点。取得 GPU 后的阶段时长包括生成、CPU offload、数据处理和 checkpoint I/O，排除资源排队；不能只累计 CUDA kernel 时间制造效率优势。数值异常或预注册 KL 阈值越界分别记为异常／保护停止，不能冒充正常预算终点。

成本报告包括对齐分支实际 GPU 占用时长、端到端分支运行时长、CPU 卸载和采样时间，以及额外的 RM 训练和 reference 缓存成本。共用父权重成本单列，再分别给“已有 SFT/RM 时的边际成本”和“从原始模型开始的端到端成本”；不以相同步数宣称等算力。

## 无新增人工标注的评测

| 评测 | 指标与范围 | 不支持的结论 |
|---|---|---|
| 封存 HelpSteer3 偏好 | RM pair accuracy／log-loss；按领域、长度差、偏好强度分层 | 新生成回答的人类胜率 |
| 策略在现成回答上的偏好拟合 | 所有策略报告 chosen NLL 与明确标注的 likelihood ranking；DPO／PPO 另报相对 SFT 的 log-ratio ranking | log-prob 高就是回答事实正确 |
| 完整 IFEval | 官方 strict／loose、prompt／instruction 四类指标；保存全部生成和判定 | 综合有用性或“零幻觉” |
| 完整 RM-Bench | 用官方脚本检查训练 RM 的正确性／风格敏感性及分域结果 | 本项目生成策略的真人质量 |
| 冻结奖励投机诊断集 | RM 分数、重复／长度、已知规则违反率、KL 和 EOS 率 | 一个通用“质量提升率” |

SFT 作为 reference 时，其 log-ratio 恒为零，不制造一个有区分力的“SFT 隐式奖励分数”。RM accuracy 与 policy likelihood ranking 分开解释；若报 DPO 隐式奖励，完整记录 beta × log-ratio 的定义。

IFEval 最终使用官方完整输入和评测脚本，统一 greedy 解码、最多 2,048 个新 token，记录到达上限的比例。基于约束文本预审长度需求，不能沿用训练的 128／256 token 上限让长回答任务必然失败。上限仍不足时如实列为截断，若修订协议则所有模型同样重评；短子集只能另命名，不能称完整官方分数。最终集不参与 checkpoint 选择。[官方 IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)

RM-Bench 只对公共训练 RM 做完整外部诊断，不额外引入大型 GPU 裁判。它考察细微内容差别和表达风格，训练数据重合和基座污染仍需披露。[官方 RM-Bench](https://github.com/THU-KEG/RM-Bench)

投机诊断集预先固定 200 个独立诊断 prompt：100 个来自独立 diagnostic-holdout 对话簇，100 个由可程序校验的约束模板生成，与调参集和其他评测集去重。加入重复段落、无关赞美、离题续写；只有带确定规则的模板才使用数字／实体破坏并称为已知错误。对同一回答的原版和扰动版做配对评分，内容无真值的扰动只称压力测试。该集不用于调参；真实 PPO 奖励上升与其他指标下降只能支持“出现分歧”，不能凭分歧断言哪一个模型绝对正确。

每个 seed 单独给配对差值，对 prompt 簇进行至少 2,000 次成组配对 bootstrap；另报三 seed 均值和样本标准差。置信区间条件于这些已训练模型，不把同一 prompt 的多个回答或多个 seed 摊平成独立样本。主要指标提前指定为 IFEval strict prompt accuracy；偏好拟合、成本与鲁棒性保持分项，不拼成可随意调权的综合分数。

## 容量验收和时间预算

先做四小时以内的独立工程校准，允许从短序列检查量化前后向，但正式容量通过必须覆盖完整目标长度、DPO 更新、PPO 采样／评分／更新、最长评测输出和断点恢复。至少连续完成 20 个 DPO optimizer updates 与 8 个 PPO rollout batches，并通过冻结参数、梯度、有限值及恢复测试。无 OOM 只是必要条件；不能用 CPU smoke 或旧 GPT-2 容量报告替代。

| 阶段 | 设计预留的本机活跃运行时间 |
|---|---:|
| 量化兼容性、容量及吞吐校准 | 至多 4 小时 |
| 公共 SFT 和 RM | 合计约 12 小时 |
| DPO／PPO 六个主实验 | 每个至多约 8 小时，合计 48 小时 |
| 四个探索消融 | 每个约 2 小时，合计 8 小时 |
| 自动评测、完整核验和报告 | 预留约 12 小时 |
| 合计设计额度 | 84 小时；正式调度额度仍待吞吐校准后冻结 |

**84 小时是设计额度，不是“84 小时必定全跑完”的预测。** 数据准备、实现和兼容性修复另计；GPU 排队另列，不能让无人值守等待无限消耗或延长额度。仅把这个额度连续用满就约 3.5 天，因此新版按数天而非几小时规划。首次吞吐数据出来后再给日历 ETA。

训练目标同时记录样本／token／optimizer-update 数。正式配置必须在校准后、看最终结果前确定：完成一遍目标数据的工作量、每个分支允许的计算量，以及全局 wall-clock 截止。估算不足时先缩减可选消融或报告额度不足，不静默降低核心数据规模、种子数或最终评测分母。最终 IFEval 生成长度可能使 12 小时评测预留不足，需按实测长度分布和 tokens/s 单独测算。

正式实现要区分按预注册计算剂量正常结束的实验与被调度截止中断的实验。预算到期、OOM、崩溃或未完成核验都不算整个项目完成；只有全部预注册主实验、必需消融和全量最终评测完毕，才能给完整结论。`data_pass_complete` 与 `protocol_complete` 分别记录，未遍历完整数据不能写“全数据训练完成”。

## 交付物与完成标准

| 交付物 | 可检查证据 |
|---|---|
| 数据卡和切分清单 | 上游 revision、许可、过滤明细、实际 pair／prompt 数、重合审计 |
| 8GB 训练系统 | NF4／LoRA、reference 缓存、真正 PPO、adapter 切换与恢复测试 |
| 可复现实验 | 独立环境锁、配置和源码哈希、父权重身份、真实日志、阶段清单 |
| 模型产物 | SFT／RM／DPO／PPO adapters、heads、tokenizer 和明确模型卡 |
| 研究报告 | 三 seed 波动、成本—表现曲线、正则化消融、负结果与失效案例 |
| 本地演示 | 同 prompt 并排比较 SFT／DPO／PPO，显示真实检查点和训练配置 |
| 简历材料 | 只从已验证报告提取规模、显存、吞吐和指标差值，连接可复算的结果 |

最终完成要求：容量和正确性验收通过；冻结的核心数据门槛满足；六个主分支与预注册消融均按协议结束；最终评测覆盖与哈希核验通过；报告能从保存的原始输出重新生成。只做到训练循环可运行，应标工程验证，不能宣称本研究完成。

可在完成后据实写入简历的描述模板：

> 独立构建面向 8GB 单卡的 1.7B 大模型偏好对齐系统，实现 QLoRA SFT、奖励建模、DPO 与 PPO；通过共享冻结基座和参考概率缓存控制显存开销。基于公开真人偏好数据完成三随机种子对照，以 IFEval、留出偏好和 RM-Bench 分析训练成本、指令遵循及奖励投机，提供可复现报告与模型演示。

上述句子描述的是交付目标，尚不能当作已完成履历。只有报告产生真实结果后，才加入“峰值显存 X、吞吐 Y、指标差值 Z”等数字；不填写预期胜率。

## 实施顺序和项目隔离

1. 收尾已取消的 GPT-2 当前子进程，保留 STOP、源码、配置和历史产物；不再恢复九组旧计划。
2. 新代码使用独立包与新配置／输出路径；先完成数据 CPU 审计和独立环境，不在其他项目的 Python 中安装依赖。
3. 原训练退出后实施容量与正确性门槛，输出实际吞吐、峰值和预算测算；冻结正式计划后才派发长训练。
4. 先主实验，再消融及最终评测；每半小时按北京时间报告真实进度、当前进程、剩余工作和故障。
5. 新 RLHF 完成或确认无法继续时，按 `reports/priority-20261020.json` 恢复 VLA 第173步断点到新目录；其退出后恢复 EviDoc GPU 推理服务及 ForgeLM 原等待队列。重设计期间保留用户指定的先后顺序，不无限维持无法推进的资源占用。

当前只有本设计和交接记录；不能把本文路径传给旧 campaign runner。执行配置应另建 `configs/alignlab-v2/`，新输出使用独立身份目录，旧 `runs/local-gpu` 与 `runs/study-v1-20261020` 均保留。
