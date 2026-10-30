# AlignLab · Qwen3-1.7B 偏好对齐实验

当前状态：V10已于2026-10-30T02:00:50.4678370Z单次派发；最新冻结monitor采样2026-10-30T02:24:37.1616235Z，状态v10_running，新PPO44进度6/541（未验收partial），正式接受仍3787条/7评分。guardian=True、awake=True、unknown=0，RAM=10.815445GiB/GPUfree=4356MiB，EviDoc恢复pending=True。两真实准入间隔321.307秒；运行遥测与准入分列。禁止重复派发或额外Python测试/模型加载，仅冻结PSmonitor。旧partial303/213排除，PPO44完整评分/两RM/主分析尚待完成，原截止及ETA=null不变。 checkpoint：C:\Users\Intern1\Documents\ChatGPT\chenxi\rlhf-lab\reports\alignlab-quality-v10-launch-20261030T013000Z\runtime-checkpoint-20261030T0225063845414Z.json；SHA256 47478285ea2e33c7fcf6580a47bd03aaef0ca98ee0b286f0afc534c6e7e194ab。

历史GPU首条生成独立复核（2026-10-29 07:29 UTC采样）：当时PPO44 inference_running 1/541，精确进程链60156→61168→49032、守护与运行资源全部通过；尚未完成PPO44完整推理/评分及主分析，旧partial303排除、人工标签0/胜率null。证据：C:\Users\Intern1\Documents\ChatGPT\chenxi\rlhf-lab\reports\alignlab-quality-v9-launch-20261029T064300Z\independent-first-generation-review-20261029T0732140249390Z.json；SHA256 37b6e57471f007bb024f26aa71750aa9156836620a51d416d0ea097bd34e3b05。

**历史阶段（已由V9失败闭环状态取代）：截至 2026-10-29 06:31:54 UTC（北京时间14:31）：V8单次人工启动尝试在首次资源准入失败，已安全恢复服务；GPU评测未启动。** 用户明确要求“那就启动啊”后，已启动新guardian并精确暂停EviDoc60124；06:30:00 UTC真实可用RAM12.018177GiB低于12.5GiB，唯一blocker为内存，GPUfree6847MiB/util7、5保留身份及服务通过、unknown=[]。未做第二准入、未创建execution-launch/process/fresh output。已立即请求恢复，EviDoc于06:30:47 UTC按原配置恢复为PID60380 parent61972/create2026-10-29T06:30:17.9429660Z/CIM ticks字符串639288522179429660；根与独立代理再次验证完整身份、8083唯一owner、28/28健康字段、5保留Python，无未知Python，guardian61972已complete并退出。恢复pending=false，原V8计划及042700分配永久失败不得重派。先前V8静态192源码/CPU121/PS93及7结果链/完整9模型容量桥通过的历史仍保留；本次未修改冻结代码或放宽阈值，未操作其他服务或应用。七组正式推理与官方评分共3787条仍有效，PPO44部分303不计入；尚缺PPO44全量、两RM、主分析及最终验收，无可靠总ETA。后续保持服务恢复并只读监护，合法接续必须新计划及60380完整恢复链，禁止循环暂停试探。原截止2026-11-03 02:30:11 UTC不变。闭环：prelaunch-admission-failure-closure-20261029T0631546353574Z.json SHA a66fef19a9d50771ff5f0bd2241da593a53b394e41c4d84c63a1ec662c329340。

以下为先前各阶段的历史记录，具体身份和运行状态仅适用于其标注时点。

AlignLab 研究单张 8 GiB GPU 上的语言模型偏好对齐：以 Qwen3-1.7B 为基座，使用 NF4 量化、LoRA、共享 SFT 与奖励模型，比较随机种子 42、43、44 的 DPO 和 PPO。仓库同时保留数据审计、模型接受、资源约束和失败追溯的实现。

**截至 2026-10-29 02:40:33 UTC：PPO44 在303/541条后失败，V6已停止并安全闭环；此前7/8个策略官方评分已接受。** 直接异常为服务保护检查发现冻结精确身份之外的Python PID58104，失败时间2026-10-29 02:34:05UTC；最后资源记录RAM12.0496GiB/GPU空闲3803MiB/allocator2340MiB均在限额内。该时点未保存PID58104完整命令、parent及create，来源尚未确认，不能按当前同号PID归因或操作。累计已写4090/4328条，其中3787条为此前完整接受的7组，PPO44的303条无最终manifest、尚未验收，不能直接继承。PPO44 exit1、campaign failed；全部42个已记录评测精确身份均退出，guardian40508 complete、防休眠41904 released后退出。EviDoc已于02:35:13UTC按原配置自动恢复，新PID55612 parent40508/create2026-10-29T02:34:38.4723310Z/ticks字符串639288380784723310；本次现场复核完整身份、8083及28/28健康字段全部通过，restore_pending=false。5保留Python身份匹配，闭环时无未知Python。7组评分实际SHA、官方标志及541prompt/834指令全分母再验通过。此前生成ETA已撤回，主分析与RM两项尚未完成；原截止2026-11-03 02:30:11UTC不变。失败V6 plan/output绝不重派，不放宽身份规则或停止其他服务；保留全部失败与已接受结果，调查进程来源并准备另行验证的合法接续。失败闭环证据：failure-closure-20261029T0240335529494Z.json。

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

2026-10-26 14:48:01 UTC 的 EviDoc 暂停检查曾短暂记录 absence_matches=false；14:49:24 UTC 同一监测脚本复核恢复正常，直接查询原 PID50784 不存在、8083 无监听。原快照没有记录失败谓词及进程/监听身份，原因未确定；冻结监测脚本的 PID-only 缺席判断可能受 PID 复用影响。全部异常和复核证据保留于 pause-absence-recheck-20261026T1451199249408Z.json；评测未重派，服务未停止/重启/恢复，冻结代码未改。

监护读取历史（2026-10-28 22:43 UTC）：一次 live JSON 解析出现 Unterminated string，monitor exit1；22:44:12 UTC 新一次冻结监护已成功，PPO44 为 165/541，累计 3952/4328，原评测进程、守护及资源正常。静态复核确认 live campaign/progress 原位写入存在读取竞争窗口；本次与该机制相符，但失败原始字节及确切读取文件未捕获，不能确证原因，也不能当作评测子进程失败。未重派推理或操作服务。观察记录：monitor-live-json-read-recovery-20261028T2246216377353Z.json。

历史接续准备（已由06:31 V8准入失败闭环取代）：新V8（2026-10-29 04:40 UTC）完整静态prepare已exit0，正式plan已发布且SHA与candidate一致；192源码（原173不变、新19）、七组完整推理及官方评分继承链和九模型完整容量桥全部实际校验通过，两个准备进程已退出。CPU第三轮121/121、PS离线93/93通过；PPO44失败部分303条仍排除，新PPO44须从0开始。06:25:45 UTC只读观察：EviDoc60124完整身份/8083唯一监听/28健康字段及5保留Python全部通过，unknown=[]，RAM10.1217GiB、GPUfree2458MiB（EviDoc运行中）；尚无足以支持新暂停的资源状态改善。ready_for_launch=false，未暂停服务、未准入、未GPU执行、未创建输出；保持服务恢复并只读等待自然资源变化，不重复prepare/测试、不循环暂停试探、不放宽门槛。原截止不变，无可靠总ETA。
