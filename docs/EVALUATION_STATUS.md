# 独立质量评测状态

当前状态：V10已于2026-10-30T02:00:50.4678370Z单次派发；最新冻结monitor采样2026-10-30T02:24:37.1616235Z，状态v10_running，新PPO44进度6/541（未验收partial），正式接受仍3787条/7评分。guardian=True、awake=True、unknown=0，RAM=10.815445GiB/GPUfree=4356MiB，EviDoc恢复pending=True。两真实准入间隔321.307秒；运行遥测与准入分列。禁止重复派发或额外Python测试/模型加载，仅冻结PSmonitor。旧partial303/213排除，PPO44完整评分/两RM/主分析尚待完成，原截止及ETA=null不变。 checkpoint：C:\Users\Intern1\Documents\ChatGPT\chenxi\rlhf-lab\reports\alignlab-quality-v10-launch-20261030T013000Z\runtime-checkpoint-20261030T0225063845414Z.json；SHA256 47478285ea2e33c7fcf6580a47bd03aaef0ca98ee0b286f0afc534c6e7e194ab。

历史GPU首条生成独立复核（2026-10-29 07:29 UTC采样）：当时PPO44 inference_running 1/541，精确进程链60156→61168→49032、守护与运行资源全部通过；尚未完成PPO44完整推理/评分及主分析，旧partial303排除、人工标签0/胜率null。证据：C:\Users\Intern1\Documents\ChatGPT\chenxi\rlhf-lab\reports\alignlab-quality-v9-launch-20261029T064300Z\independent-first-generation-review-20261029T0732140249390Z.json；SHA256 37b6e57471f007bb024f26aa71750aa9156836620a51d416d0ea097bd34e3b05。

**历史阶段（已由V9失败闭环状态取代）：截至 2026-10-29 06:31:54 UTC（北京时间14:31）：V8单次人工启动尝试在首次资源准入失败，已安全恢复服务；GPU评测未启动。** 用户明确要求“那就启动啊”后，已启动新guardian并精确暂停EviDoc60124；06:30:00 UTC真实可用RAM12.018177GiB低于12.5GiB，唯一blocker为内存，GPUfree6847MiB/util7、5保留身份及服务通过、unknown=[]。未做第二准入、未创建execution-launch/process/fresh output。已立即请求恢复，EviDoc于06:30:47 UTC按原配置恢复为PID60380 parent61972/create2026-10-29T06:30:17.9429660Z/CIM ticks字符串639288522179429660；根与独立代理再次验证完整身份、8083唯一owner、28/28健康字段、5保留Python，无未知Python，guardian61972已complete并退出。恢复pending=false，原V8计划及042700分配永久失败不得重派。先前V8静态192源码/CPU121/PS93及7结果链/完整9模型容量桥通过的历史仍保留；本次未修改冻结代码或放宽阈值，未操作其他服务或应用。七组正式推理与官方评分共3787条仍有效，PPO44部分303不计入；尚缺PPO44全量、两RM、主分析及最终验收，无可靠总ETA。后续保持服务恢复并只读监护，合法接续必须新计划及60380完整恢复链，禁止循环暂停试探。原截止2026-11-03 02:30:11 UTC不变。闭环：prelaunch-admission-failure-closure-20261029T0631546353574Z.json SHA a66fef19a9d50771ff5f0bd2241da593a53b394e41c4d84c63a1ec662c329340。

以下为先前各阶段的历史记录，具体身份和运行状态仅适用于其标注时点。

**截至 2026-10-29 02:40:33 UTC：PPO44 在303/541条后失败，V6已停止并安全闭环；此前7/8个策略官方评分已接受。** 直接异常为服务保护检查发现冻结精确身份之外的Python PID58104，失败时间2026-10-29 02:34:05UTC；最后资源记录RAM12.0496GiB/GPU空闲3803MiB/allocator2340MiB均在限额内。该时点未保存PID58104完整命令、parent及create，来源尚未确认，不能按当前同号PID归因或操作。累计已写4090/4328条，其中3787条为此前完整接受的7组，PPO44的303条无最终manifest、尚未验收，不能直接继承。PPO44 exit1、campaign failed；全部42个已记录评测精确身份均退出，guardian40508 complete、防休眠41904 released后退出。EviDoc已于02:35:13UTC按原配置自动恢复，新PID55612 parent40508/create2026-10-29T02:34:38.4723310Z/ticks字符串639288380784723310；本次现场复核完整身份、8083及28/28健康字段全部通过，restore_pending=false。5保留Python身份匹配，闭环时无未知Python。7组评分实际SHA、官方标志及541prompt/834指令全分母再验通过。此前生成ETA已撤回，主分析与RM两项尚未完成；原截止2026-11-03 02:30:11UTC不变。失败V6 plan/output绝不重派，不放宽身份规则或停止其他服务；保留全部失败与已接受结果，调查进程来源并准备另行验证的合法接续。失败闭环证据：failure-closure-20261029T0240335529494Z.json。

## V5 失败与接续准备历史（原文保留）

状态快照：2026-10-26 08:37:11 UTC。本文是本地证据的发布摘要，不是完整运行日志，也不是质量结果报告。

## 当前结论

主训练成功完成，共享 SFT/RM 与六组 DPO/PPO 产物已接受，相关训练已退出。V5 九模型完整容量 **9/9** 已获监督进程验收，`campaign.capacity_verified=true`；容量任务及 `ifeval-base` 正式推理均已 `accepted/exit0`，base 完整 **541/541** 条，总生成进度 **541/4,328（12.5%）**。随后 `ifeval-base-score` 以退出码 **1** 失败，campaign 为 `failed`：NLTK 前置校验将 77 项声明 pins 与 4 项英文必需 pins 作整字典比较，错误拒绝有效超集。评分算法尚未进入，这次失败不来自 GPU 或模型，尚无官方质量分数，主分析未开始；完整独立质量评测仍未完成。

正式评测生成数为 **541/4,328**（base 为 **541/541**），已经接受的生成结果保留，但不能替代官方评分或完整分母验收。容量试运行输出不计入正式评测分母。人工标签数为 0，人工胜率为 `null`；奖励值不等于独立质量提升。

08:37:11 UTC 失败收尾复核：全部 **8 个评测 PID**、恢复守护 PID 33572 和防休眠 PID 50212 已退出；guardian 为 `complete`，awake 为 `released`。EviDoc 已按恢复 receipt 启动为 PID **50784**、父 PID **33572**，创建时间 **2026-10-26T08:30:24.4943760Z**、UTC ticks **639286002244943760**；根线程现场复核精确身份、8083 监听归属与原 **28 项健康字段**全部通过。五个保留 Python 匹配，无未知 Python；`restoration_complete=true`、`restore_pending=false`。收尾证据 `reports/alignlab-quality-v5-launch-20261026T040000Z/failure-closure-20261026T0837115081258Z.json` SHA-256 `e269e155d04703067a78bba5f20b2fd0890e781c423370e57a912e7e40dc26fb`；同目录 `restoration-complete.json` SHA-256 `1483c6040836af8eac90fceecb232f8a222fb7baf3938d54af38832d2dbbc259`。本次失败已完成进程与服务收尾，不代表评测成功。

后续正准备仅修复评分校验、显式继承已接受 base 541 条结果的全新 **V6**，目前**未派发**；旧 V5 计划与输出禁止原样重派，原绝对截止 **2026-11-03 02:30:11 UTC**（`1793673011`）不变。首 50/100 条观察及下方全部历史保持原样；此前 **17–40 小时**仅是当时生成阶段的条件估算，当前已失效为完成时间预测，不可当作倒计时，当前总 ETA 未知。

V4 失败后相关评测进程已退出，临时资源调整已恢复并验证。V5 修复已实现，并通过 95 项当前 CPU 检查：0 失败、错误、跳过，源码未变且 CUDA 未初始化。新计划冻结 129 个源码文件，原 107 个未改。2026-10-26 03:54 UTC，恢复守护先启动后，已按授权临时暂停 EviDoc；完成两次间隔至少五分钟的资源准入后，于 04:01:09 UTC 单次启动 V5。启动时的 `monitor-20261026T0401290599003Z.json` 确认实际监督进程 PID 49604（父 PID 51244）、恢复守护 PID 33572 和防休眠辅助进程 PID 50212 身份匹配且健康；EviDoc 处于授权临时暂停状态，ShopAssist 健康。04:01:29 UTC 的历史快照仍在冻结预检，campaign 尚未生成、进度列表为空，当时本版 CUDA 容量和正式推理尚未开始。原失败计划与输出不会原样循环重派，也不会重新训练来补时长。

V5 只替换奖励模型实例的注意力计算：显式展开 GQA 的键和值，实际支持时使用 Efficient SDPA，否则以 128 个查询位置为一块、保留全部键和值及绝对因果位置计算。保持原 FP32 归一化与 RoPE 后的 BF16 autocast 边界，参数、输入长度、padding、NF4/BF16 和 2,560 MiB 上限不变；策略模型及全局注意力注册不变。CPU 数值对照覆盖完整因果、右 padding、跨块位置、GQA 和真实混合精度路径。

04:17:24 UTC 的 `monitor-20261026T0417240806286Z.json` 已记录 RM 两个最长完整输入：HelpSteer3 **5,481 token**、RM-Bench **2,363 token**，均在 `cuda:0` 以 BF16 完成。每个输入实际执行 28 次 `torch_efficient_sdpa`，合计 **56 次**，没有使用查询分块 math 分支；全部键和值列保留，输入未截断。RM 峰值 reserved 显存为 **2,401,239,040 bytes（2,290 MiB）**，低于 2,560 MiB 硬上限；前后完整 frozen audits 一致且 `unchanged=true`。这是本次实际 CUDA 证据，不继承 V4 的 2/9，也不等于九模型容量或正式评测全部完成。

同一监测确认四个评测进程的精确身份全部匹配，恢复守护及防休眠健康。`resource-observation-20261026T0417245706836Z.json` 记录保留的五个 Python 进程身份全部匹配、无未知 Python；系统可用 RAM 约 **13.0207 GiB**，整卡空闲显存 **6,440 MiB**。这些是该时点观测，不能代替后续持续资源监护。

04:30:49 UTC 的 `monitor-20261026T0430490310008Z.json` 新增 PPO42 的两项完整容量记录：同一份 **373-token** 最长 prompt 未截断，自然生成 **92 token** 后真实 EOS 结束；强制压力检查完整生成 **2,048 token**，最终 KV 缓存覆盖 **2,421 token**，末动作已写入缓存。两项均为容量检查输出，不计入正式质量样本。PPO42 前后两次完整 frozen audits 一致且 `unchanged=true`；该条目记录的峰值 reserved 显存为 **2,401,239,040 bytes（2,290 MiB）**，低于 2,560 MiB 硬上限。同期四个评测进程、恢复守护、防休眠及其余五个常驻 Python 身份和状态正常，无未知 Python；资源观测记录可用 RAM **13.1336 GiB**、整卡空闲显存 **4,160 MiB**。正式生成仍为 **0/4,328**，评分未开始。

04:56:52 UTC 的 `monitor-20261026T0456523117985Z.json` 确认监督进程已验收九模型容量，容量子进程 `accepted/exit0`，容量 manifest SHA-256 仍为 `db2758079851ba2181426dbc506307ad234fa9a657bebd5eec30137d90aee0ed`。新 `ifeval-base` 任务处于 `dispatching`：包装进程 PID 50316（父 PID 49604）、实际推理进程 PID 53308（父 PID 50316）均存在且完整身份匹配，尚未记录正式生成结果，仍为 **0/4,328**。监督进程 PID 49604 及其包装进程 PID 51244、恢复守护 PID 33572、防休眠 PID 50212 正常；五个保留 Python 身份匹配，无未知 Python。同期资源观测可用 RAM **15.483261 GiB**、整卡空闲显存 **7,374 MiB**。这些观测证明容量验收和正式任务派发，不证明已有正式质量结果。

05:02:52 UTC 的 `monitor-20261026T0502520422667Z.json` 记录首批正式产出：`ifeval-base` 进度为 `inference_running`、`completed_records=7`。九模型容量继续保持 `capacity_verified=true`，旧容量任务已 `accepted/exit0`；当前监督进程 51244/49604 与正式推理进程 50316/53308 的完整身份均匹配。恢复守护 PID 33572 与防休眠 PID 50212 健康，五个保留 Python 身份匹配，无未知 Python；同期可用 RAM **13.016365 GiB**、整卡空闲显存 **4,007 MiB**。EviDoc 仍处于授权临时暂停状态，`restore_pending=true`，恢复责任未解除。官方评分与主分析未开始；首批 **50–100** 条正式记录积累后才按实际吞吐估算剩余时间，不以容量测试速度推算完成时间。

## 三次失败

| 尝试 | UTC 时间 | 已观察到的原因 | 完整容量 | 正式评测 |
| --- | --- | --- | --- | --- |
| V2 | 2026-10-25 09:07:32 | 首个 base 模型加载时临时申请约 1.16 GiB，触及 2.5 GiB 进程 allocator 上限；整卡尚有空闲显存 | 0/9 | 未开始 |
| V3 | 2026-10-25 09:41:30 | 加载修复后完成首个模型加载和自然 CUDA 生成，但可用系统 RAM 降至约 7.176 GiB，低于 8 GiB 运行下限；该自然生成未验收 | 0/9 | 未开始 |
| V4 | 2026-10-26 03:05 | 获准的临时资源调整解决了前次 RAM 障碍；RM 的 5,481-token 完整输入进入 SDPA 时临时申请约 1.79 GiB，超过 2.5 GiB allocator 余量 | 2/9 | 未开始 |

V4 失败前系统 RAM 仍可用约 13.355 GiB；错误发生在奖励模型长上下文注意力计算，而非训练产物接受或数据缺失。当时的静态诊断指向注意力临时显存峰值，尚未证明替代实现能通过该检查；上述 V5 实际运行随后通过了 RM 两个最长输入的容量检查。

V4 源码对应的 77 项 CPU 检查通过，模型加载实现此前也有独立 CPU 证明。这些证明各自绑定当时的源码；V5 新增实现另由当前 95 项 CPU 证明覆盖，仍不能替代九模型真实 CUDA 容量验收。V5 首次静态准备发现三处时间戳文本末尾的零被序列化省略，已按原记录修正配置，原配置及失败证据保留；未改代码或放宽身份校验。

## 固定范围与截止

- 九模型：base、SFT、RM、DPO42/PPO42、DPO43/PPO43、DPO44/PPO44。
- IFEval：八个策略模型 × 541 prompt，每个模型 834 条指令，共 4,328 次生成。
- RM：HelpSteer3 留出 2,017 对，RM-Bench 1,327 条记录；全分母保留。
- greedy、thinking=false、max_new_tokens=2048、不截断；官方 scorer seed=0、bootstrap=2000。
- NF4、原计算精度、单 batch、2,560 MiB allocator 硬上限；256 MiB 上下文余量，整卡持续空闲至少 1,024 MiB，系统可用 RAM 至少 8 GiB。

原独立评测窗口从 **2026-10-25 02:30:11 UTC** 开始，绝对截止为 **2026-11-03 02:30:11 UTC**：216 小时、777,600 秒，`started_at_epoch=1792895411`、`deadline_epoch=1793673011`。该有限上限由助手在既有执行授权范围内选择，不是用户亲自给出的数字。准备、验证、资源等待、失败尝试、推理、评分和分析均计入同一窗口；恢复版本不重置或自动延长截止，提前完成即结束。

只有完整正式推理、官方评分、完整分母、参数只读核验及主分析验收全部通过，且相关进程退出，才能报告评测完成。失败、停止或预算耗尽均不构成成功。

## 本地证据指纹

以下只发布证据的 basename 和 SHA-256；原始日志、报告、输入数据与运行目录未上传。哈希便于拿到原文件后核对字节身份，本身不替代文件内容或独立验收。

| 事项 | 本地证据 basename | SHA-256 |
| --- | --- | --- |
| 主训练最终成功监测 | `monitor-20261025T0230110853484Z.json` | `ba13a4027cf18d65ac31dfc8e4201eaf46abfc71f7afbe0ecca17f2246faf33e` |
| 四组继承训练最终复验 | `inherited-prefix-final-acceptance.json` | `e3daf7039a6086478aa56e33a54eb84c72897ae911f6857df3b5576786946eaf` |
| V2 原失败收尾 | `failure-closure-20261025T0913185529251Z.json` | `81d625ed4c908de57da29c4aedf87c182c14d5fc17552a2b0f7c05f4287d101b` |
| V2 后续计数纠正；原证据保留 | `failure-closure-count-corrected-20261025T091930Z.json` | `8bf7f00cd1d09c0973cc2ad14fd128fbed8e742c460d8ead8702625488ab0d9e` |
| V3 RAM 下限失败收尾 | `failure-closure-20261025T0949007373566Z.json` | `3418f6304e0b21606172a16018690c9b938adce33e79ad22e6b10171a5a61dfa` |
| V4 RM 注意力 OOM 收尾 | `failure-closure-20261026T0316181094032Z.json` | `789f876f1a6d96bbb121bfc31ce84adede9f5989afb5868e32259db64c5bbee6` |
| V4 资源恢复完成 | `restoration-complete.json` | `0bd01b94c74dbb5c0a5c66e84d5338096f78934021a03efdae0cba34f2ca9ba0` |
| 原授权及有限预算 | `authorization-and-budget.json` | `26921633e3ef03e46d22286d8930939c28379bbd6ccf6f5a3791a90e285489bc` |
| V4 最终 CPU 检查，77 项通过 | `report.json`（2026-10-26 02:41 UTC 批次） | `37b7bed1eb5a19fd86dd16630a3485f1f0032e2ca285d11baed76ea6bc4c784b` |
| V5 当前冻结计划 | `plan.json`（`alignlab-quality-v5-launch-20261026T040000Z`） | `00b8a18afb5d75e5e6355a673bfcb4751bf1e0bbf53bf056ae07a5683fd6b2c9` |
| V5 已启动、冻结预检监测 | `monitor-20261026T0401290599003Z.json` | `6d4ae2593a77bbea119dfcfd5c42a8d2a3fb48406e6f41af504a7e9495c66b7f` |
| V5 容量 3/9、RM 最长输入通过监测 | `monitor-20261026T0417240806286Z.json` | `d5469970b6dfd0475a895fa29f2b1b83bbf6b52661d69283926688396d360acc` |
| V5 同时点只读资源观测 | `resource-observation-20261026T0417245706836Z.json` | `744a00ccfcc290e35d6830d594d099d9c6c8d8562ba909478326df7d13516b99` |
| V5 容量 5/9、PPO42 两项容量通过监测 | `monitor-20261026T0430490310008Z.json` | `7e9fa1f622b3a9f8912ef32921527514abcdf38ecd4ab04cedb827fb67bf1b22` |
| V5 04:30:49 只读资源观测 | `resource-observation-20261026T0430495210252Z.json` | `4c71f81d77c2d5f60061ba3cd3f0963d5fb05b16c8dab1a3891ea2c1cd3f8e1e` |
| V5 九模型容量已验收、ifeval-base 已派发监测 | `monitor-20261026T0456523117985Z.json` | `0c0a70ada1bace0f39c8d325a689db72ae0688b7a5ab05bbdfaacb04b2286353` |
| V5 04:56:52 只读资源观测 | `resource-observation-20261026T0456528779386Z.json` | `2b5e9769eb4076279a80b88bf6c1dead3698126a61be39650a6d767f8f5711b0` |
| V5 首批正式生成 7/4,328 监测 | `monitor-20261026T0502520422667Z.json` | `fc9ad9a809bdfb36ef413ec2158021d3c6beaad728f93c1e282f2394bb70adfa` |
| V5 05:02:52 只读资源观测 | `resource-observation-20261026T0502526060961Z.json` | `b7bb2947d7dd71e5874b3085ce0f7209ce351741c618cb444098dfa2428c2bc6` |

历史设计见 [ALIGNLAB_V2.md](ALIGNLAB_V2.md)，发布范围见 [PUBLISHING.md](PUBLISHING.md)。历史设计中的预算、容量和进度描述不替代本文的状态快照。


04:22:45 UTC 新监测记录容量进度 4/9，新增 DPO42。全部四个评测进程、恢复守护和防休眠正常，其余五个常驻 Python 完整身份匹配，无未知 Python；可用 RAM 12.893 GiB、GPU 空闲 6,395 MiB。正式生成仍为 0/4,328，官方评分未开始。证据为 `monitor-20261026T0422459544792Z.json`（SHA-256 `a45786a3a0760e296b1584ccd65504b9148f207b921606c04a9f8ffccb368ae0`）及 `resource-observation-20261026T0422464394279Z.json`（SHA-256 `a6b04df4ff7dde3e507a8846243d985df6552af217608c2b856462527c0e12d4`）。


04:35:48 UTC：容量进度 **6/9**，新增 DPO43；其自然生成与强制 2,048-token 检查均已记录，峰值 reserved 2,290 MiB，前后完整参数审计一致。仍有 PPO43、DPO44、PPO44 待完成；正式生成 0/4,328，评分未开始。四个评测进程、恢复守护、防休眠及五个保留 Python 身份正常，无未知 Python；可用 RAM 12.926 GiB、GPU 空闲 4,110 MiB。证据 `monitor-20261026T0435481791821Z.json`，SHA-256 `16d123e807c7ccc4a5a0a797cfa8758d88184ea33a4f1f9b9354a027bd20dfa7`；资源证据 `resource-observation-20261026T0435486577573Z.json`，SHA-256 `c069733189293f97c723759a1cdd831c4d73e0aa4e24ea08a5e08420fd2b1f3b`。


04:40:49 UTC：容量进度 **7/9**，新增 PPO43；自然 EOS 与强制 2,048-token 检查均已记录，显存峰值 reserved 2,290 MiB，前后完整参数审计一致。剩余 DPO44、PPO44；正式生成 0/4,328，评分未开始。四个评测进程、恢复守护、防休眠及五个保留 Python 身份正常，无未知 Python；可用 RAM 12.731 GiB、GPU 空闲 4,043 MiB。证据 `monitor-20261026T0440495794392Z.json`，SHA-256 `318888b99b29e48e3f82e5622e5bbe3a5b54cfea9a49de0aa335e3cbb9ad6902`；资源证据 `resource-observation-20261026T0440500938143Z.json`，SHA-256 `a91c987e51b4e4f7ec59ca76b9f814873cf318f50f9af22cb3508237e6c7036c`。


04:45:50 UTC：容量进度 **8/9**，新增 DPO44；自然 EOS 与强制 2,048-token 检查均已记录，显存峰值 reserved 2,290 MiB，前后完整参数审计一致。仅剩 PPO44；正式生成 0/4,328，评分未开始。四个评测进程、恢复守护、防休眠及五个保留 Python 身份正常，无未知 Python；可用 RAM 12.562 GiB、GPU 空闲 4,042 MiB。证据 `monitor-20261026T0445505765418Z.json`，SHA-256 `8ce8f3a0f70fd5e12decd1458e2d3babe228f2ffa9d09733391ca49c74a111f0`；资源证据 `resource-observation-20261026T0445511029724Z.json`，SHA-256 `ad505a7342a236fbda959b3f54b2e6047d1396a7afe145c0f5d3f940932aa8f4`。


04:50:47 UTC：九模型容量子任务 **9/9** 完成，PPO44 自然 EOS 生成 92 token，压力检查完整生成 2,048 token。八个 policy 两类测试和 RM 两个最长完整输入均有实际 CUDA 记录；全部 18 份完整参数审计未变化，最高 reserved 2,290 MiB。容量子进程已正常退出，监督进程仍在验收，正式生成 0/4,328。可用 RAM 15.541 GiB、GPU 空闲 7,363 MiB；监督进程、恢复守护、防休眠及五个保留 Python 正常，无未知 Python。监测 `monitor-20261026T0450476714377Z.json` SHA-256 `4fadef8ee0f2ac6a0d2355f6b861da686eed7e590149c98de8aa164c903f424e`；资源证据 `resource-observation-20261026T0450481532290Z.json` SHA-256 `77649d83e29a31b8010051efe6f37139cb456e98c4466b13e242045851e4b5b5`；容量 manifest SHA-256 `db2758079851ba2181426dbc506307ad234fa9a657bebd5eec30137d90aee0ed`；独立只读复核里程碑 `nine-model-capacity-child-passed-20261026T045047Z.json` SHA-256 `a7fea7abffcf3c97b871f4ae026afd422e6542eaf70b59980aa798733418dfd6`。

05:07:48 UTC：`ifeval-base` 持续 `inference_running`，正式生成 **28/541**，总计 **28/4,328**；九模型容量 `capacity_verified=true`，旧容量任务 `accepted/exit0` 后进程缺席正常。四个活跃评测进程、恢复守护、防休眠和五个保留 Python 身份及状态正常，无未知 Python；可用 RAM **12.428970 GiB**（高于 **8 GiB** 运行下限）、GPU 空闲 **4,010 MiB**。EviDoc 仍按授权临时暂停且等待恢复；官方评分与主分析未开始，正式记录不足 50 条，暂不估算 ETA。监测 `monitor-20261026T0507487615628Z.json` SHA-256 `8b1e2016e1dcaab99b0dce492bb424585ab5e81b89edc202cba19e8880ab91fa`；资源观测 `resource-observation-20261026T0507493184340Z.json` SHA-256 `27e34ded2a86f169ceb0862b1806dcc88ee65fea8609546b8fd3250133ad24db`。

05:12:50 UTC：`ifeval-base` 持续 `inference_running`，正式生成 **37/541**，总计 **37/4,328**；九模型容量 `capacity_verified=true`。四个活跃评测进程、恢复守护、防休眠和五个保留 Python 身份及状态正常，无未知 Python；可用 RAM **12.431946 GiB**、GPU 空闲 **4,012 MiB**。官方评分与主分析未开始，正式记录不足 50 条，暂不估算 ETA。监测 `monitor-20261026T0512504373817Z.json` SHA-256 `b7ef02e6438b1b926e78d584e17fb6c301dfa9c7ea6274f12e01223313330c9b`；资源观测 `resource-observation-20261026T0512510379090Z.json` SHA-256 `d5b606cc1eb8dcf3d109203ac90af493334889297bf24b445ea3032da73c7889`。

05:17:48 UTC：`ifeval-base` 持续 `inference_running`，正式生成 **49/541**，总计 **49/4,328**；九模型容量 `capacity_verified=true`。四个活跃评测进程、恢复守护、防休眠和五个保留 Python 身份及状态正常，无未知 Python；可用 RAM **12.436565 GiB**、GPU 空闲 **4,014 MiB**。EviDoc 仍按授权临时暂停；官方评分与主分析未开始。此保存快照正式记录仍不足 50 条，暂不估算 ETA。监测 `monitor-20261026T0517485684123Z.json` SHA-256 `d91c78f60278708c010447fb309c572590d5db5d4f133ec3909c17cfb70baebf`；资源观测 `resource-observation-20261026T0517491307952Z.json` SHA-256 `21de51954014d2cda7dc80efa34bedfc794e8b12c005957f7a95b942714207e2`。

05:18:36 UTC：只读核对已写入的正式输出确认 **50 条完整结果**（此前 05:17:48 监护快照为 49 条）。前 50 条共有 18,775 个真实生成 token，均值 375.5、中位数 256、P90 823，范围 11–2,048；47 条自然 EOS，3 条达到生成上限，输入全部未截断。输出不含逐条时间戳。由阶段开始 04:58:11.518 UTC 至恰好 50 条时的最后写入 05:18:30.743 UTC，共 1,219.225 秒，阶段均速 24.38 秒/条，包含初始化、检查和写入，不能解释为纯 GPU 生成速度。若余下 base 样本与此前相近，491 条约需 3 小时 20 分钟。若所有策略后续速度也相近，4,278 条剩余生成点估约 29 小时；按三个已存监护窗口的速度变化作条件外推约 **17–40 小时**。这仅是生成阶段情景估算，不是置信区间或总完成 ETA；其他模型与剩余 prompt 的代表性未验证，后续加载、RM 推理、官方评分、主分析耗时仍未知。没有额外 benchmark 或模型加载。证据 `formal-first50-throughput-observation-20261026T0521402130051Z.json` SHA-256 `1efda8d5ad600edc8df69534c9a52eeadac33d5a3d35b6d937a0f308652a5a5b`；前 50 行以 UTF-8/LF 序列化指纹 `e4e8c04c89b37c352b4c758bcb6efb8541c6959f679cf063326026638a101b2a`。

05:22:49 UTC：`ifeval-base` 持续 `inference_running`，正式生成 **63/541**，总计 **63/4,328**；九模型容量 `capacity_verified=true`。四个活跃评测进程、恢复守护、防休眠和五个保留 Python 身份及状态正常，无未知 Python；可用 RAM **12.374466 GiB**、GPU 空闲 **4,017 MiB**。EviDoc 仍按授权临时暂停；官方评分与主分析未开始。本轮未重算首 50 条的生成阶段条件估算，仍无总完成 ETA。监测 `monitor-20261026T0522497822645Z.json` SHA-256 `81fe8d980388947aa363c38cf9521186bb7ecad3bd80760047d0d33385673930`；资源观测 `resource-observation-20261026T0522503992124Z.json` SHA-256 `523beb176eb69ae7a9a0f9b02201ab8a699e336820055c51c6290b9eef40b510`。

05:27:47 UTC：`ifeval-base` 持续 `inference_running`，正式生成 **73/541**，总计 **73/4,328**；九模型容量 `capacity_verified=true`。四个活跃评测进程、恢复守护、防休眠和五个保留 Python 身份及状态正常，无未知 Python；可用 RAM **12.228500 GiB**、GPU 空闲 **4,019 MiB**。EviDoc 仍按授权临时暂停；官方评分与主分析未开始。首 50 条生成阶段的 **17–40 小时**条件估算保留，本轮未重算，等待至少 100 条正式记录后再更新；仍无总完成 ETA。监测 `monitor-20261026T0527473642067Z.json` SHA-256 `04b8605a862164b7a116a7fb0642da1119bc1e5d9b5393d4099fb37d96d27b91`；资源观测 `resource-observation-20261026T0527479179584Z.json` SHA-256 `a87a9a1f0ca0184992d52bef3b31386aed82325aff94955a46d94e7596bfbf58`。

05:32:48 UTC：`ifeval-base` 持续 `inference_running`，正式生成 **87/541**，总计 **87/4,328**；九模型容量 `capacity_verified=true`。四个活跃评测进程、恢复守护、防休眠和五个保留 Python 身份及状态正常，无未知 Python；可用 RAM **12.256969 GiB**、GPU 空闲 **4,022 MiB**。EviDoc 仍按授权临时暂停；官方评分与主分析未开始。首 50 条生成阶段的 **17–40 小时**条件估算保留，本轮未重算，等待至少 100 条正式记录后再更新；仍无总完成 ETA。监测 `monitor-20261026T0532483767043Z.json` SHA-256 `c0a25bb08d05e2fd952b5ae64e914b4a72b7e984ab5981edb022640376bd32b1`；资源观测 `resource-observation-20261026T0532489651038Z.json` SHA-256 `788801413e2540d169740dba8852277d04a70098ef30aba104fafa5a9ddcbdd8`。

05:37:49 UTC：`ifeval-base` 持续 `inference_running`，正式生成 **99/541**，总计 **99/4,328**；九模型容量 `capacity_verified=true`。四个活跃评测进程、恢复守护、防休眠和五个保留 Python 身份及状态正常，无未知 Python；可用 RAM **12.230728 GiB**、GPU 空闲 **4,024 MiB**。EviDoc 仍按授权临时暂停；官方评分与主分析未开始。本次保存快照仍不足 100 条，保留首 50 条生成阶段的 **17–40 小时**条件估算，不重算 ETA，也不混入未确认的后续进度。监测 `monitor-20261026T0537499870084Z.json` SHA-256 `a807bf01837ccc89589c9a840571e6a5d183ea8dcc238cc3fd992b92b67a89bc`；资源观测 `resource-observation-20261026T0537505522707Z.json` SHA-256 `6073936d23c8ab23a757e4d9b9098feab7525e0d15d6cfbc405f710025007a73`。

05:38:44 UTC：现有进度已写 **101 条**；随后只读核对前 **100 条完整正式结果**，全部输入未截断。实际生成 token 共 37,938，均值 379.38、中位数 254、P90 823，范围 11–2,048；95 条自然 EOS，5 条达到生成上限。阶段开始至本次进度读取共 2,432.706 秒，101 条对应约 **24.09 秒/条**，包含初始化、检查、写入和可能在途样本；无逐条时间戳，第 100 条确切完成时刻未知，没有使用文件修改时间冒充。与首 50 条的 24.38 秒/条接近，保持生成阶段 **17–40 小时条件范围**，不强行缩窄；该历史范围不是随当前时刻倒计时的承诺，跨模型与剩余样本代表性仍未验证，RM、后续加载、官方评分及主分析总耗时仍未知。首 50 条报告完整保留。首 100 条报告 `formal-first100-throughput-observation-20261026T0540021640231Z.json` SHA-256 `80bb10d50f6c2b19625df6c2fbea8ee903dcb93b5a57769159eb4096e46079b4`；规范 UTF-8/LF 前 100 行 SHA-256 `75a21e045784d210aafda0d20639ab3c35b68ab34dc21ddeed7c5308eb312299`。

05:42:48 UTC：`ifeval-base` 持续 `inference_running`，正式生成 **117/541**，总计 **117/4,328**；九模型容量 `capacity_verified=true`。四个活跃评测进程、恢复守护、防休眠和五个保留 Python 身份及状态正常，无未知 Python；可用 RAM **12.334282 GiB**、GPU 空闲 **4,016 MiB**。EviDoc 仍按授权临时暂停；官方评分与主分析未开始。首 100 条统计、首 50 条历史和生成阶段 **17–40 小时**条件估算保持不变，本轮不重算 ETA，待 base 全部 541 条完成或吞吐明显变化后再更新。监测 `monitor-20261026T0542485694223Z.json` SHA-256 `49a76a24db4af5344465f278d00bd6b681189d629c8cac8963ce880a46bfc03a`；资源观测 `resource-observation-20261026T0542491897686Z.json` SHA-256 `f3ee5991326f26afb62f2f8cbd1ced7de7e9dddd7253e041e87cbc12632b6bee`。

2026-10-26 14:48:01 UTC 的 EviDoc 暂停检查曾短暂记录 absence_matches=false；14:49:24 UTC 同一监测脚本复核恢复正常，直接查询原 PID50784 不存在、8083 无监听。原快照没有记录失败谓词及进程/监听身份，原因未确定；冻结监测脚本的 PID-only 缺席判断可能受 PID 复用影响。全部异常和复核证据保留于 pause-absence-recheck-20261026T1451199249408Z.json；评测未重派，服务未停止/重启/恢复，冻结代码未改。

监护读取历史（2026-10-28 22:43 UTC）：一次 live JSON 解析出现 Unterminated string，monitor exit1；22:44:12 UTC 新一次冻结监护已成功，PPO44 为 165/541，累计 3952/4328，原评测进程、守护及资源正常。静态复核确认 live campaign/progress 原位写入存在读取竞争窗口；本次与该机制相符，但失败原始字节及确切读取文件未捕获，不能确证原因，也不能当作评测子进程失败。未重派推理或操作服务。观察记录：monitor-live-json-read-recovery-20261028T2246216377353Z.json。

历史接续准备（已由06:31 V8准入失败闭环取代）：新V8（2026-10-29 04:40 UTC）完整静态prepare已exit0，正式plan已发布且SHA与candidate一致；192源码（原173不变、新19）、七组完整推理及官方评分继承链和九模型完整容量桥全部实际校验通过，两个准备进程已退出。CPU第三轮121/121、PS离线93/93通过；PPO44失败部分303条仍排除，新PPO44须从0开始。06:25:45 UTC只读观察：EviDoc60124完整身份/8083唯一监听/28健康字段及5保留Python全部通过，unknown=[]，RAM10.1217GiB、GPUfree2458MiB（EviDoc运行中）；尚无足以支持新暂停的资源状态改善。ready_for_launch=false，未暂停服务、未准入、未GPU执行、未创建输出；保持服务恢复并只读等待自然资源变化，不重复prepare/测试、不循环暂停试探、不放宽门槛。原截止不变，无可靠总ETA。
