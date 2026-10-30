# 草案：双层进化、受控环境绑定与独立验证

状态：设计稿，未实现。本 PR 只交付设计和实施验收清单，供后续代码 PR 使用。
代码核对基线：`9a23c1b08b172fcbe32acea6526eaa6b428b90b9`（2026-10-30 main）。
环境定义、坐标、EnvCell 和工具兼容规则沿用 [environment-cells.md](environment-cells.md)，不另建一套。
本文中的接口名是拟议契约，除“当前实现”明确列出的能力外，不表示已经可用。

## 1. 目标与边界

OpenRSI 应能改进 agent，也能研究怎样改进“产生任务、评估、解释失败”的过程。
两者必须分层：一次受控实验内固定比较条件，跨实验周期才允许提升外围组件版本。
先打通有证据的内循环，再逐项开放外循环，不要求第一天让所有组件都自进化。

| 层 | 可改对象 | 验证依据 |
|---|---|---|
| 内循环 | prompt、memory、skill、harness；可选 weights | 冻结 judge、rubric、环境和预算下的配对 holdout |
| 外循环 | 搜索环境/场景选择、judge/eval、rubric、取证及归因方法 | 独立 anchor、固定验证环境、不可变原始记录、预注册指标 |
| 不可由候选改写 | 原始任务/轨迹/证据、独立标签、安全边界、准入规则、预算账本、sealed 集 | 可信编排方控制，任何越权使验证失败 |

外循环是版本化组件提案与验证，不是任意修改仓库、权限、奖励函数或自我打分。
weights 仅复用既有适配能力；本设计不要求新增训练后端或真实权重实验。
普通成本/时长限制与对抗安全隔离分别建模，降低预算不能替代隔离。

## 2. 当前实现与缺口

- `runtime/openrsi_runtime/loop.py` 已实现 ask → train → attribution → propose → screen → guard → apply → holdout → gate → lineage → tell。
- `ProposalRequest.attributions` 已传给提案器；`components/llm/openrsi_llm/__init__.py` 中 PromptRewriter 目前只读失败题及 rubric 失败说明，不消费 attributions。
- `EvolvingJudge` 已把 rubric 存成可补丁/快照/回滚的条目；runtime 用 `judge_every` 交替修改 agent 和 judge，并以 anchor 一致率准入 judge。
- 当前 `judge_epoch` / `progress = 2 * epoch + holdout` 是搜索排序机制，不能当跨 judge 版本的能力提升证据。
- `RunConfig` 尚无 `cells`；`score()` 的逐题字典按 task id 建键，多格同题会覆盖。
- 环境格仍是草案；描述一个格不等于执行器真的在该环境执行。
- 当前有取证、归因、版本谱系，但尚不是本文要求的完整、不可变、跨外围版本可审计证据链。

现有 judge 共进化保留为 legacy 实验路径；新严格路径不得把旧 anchor 校准等同独立外循环验证。

## 3. 两个循环的控制流

```text
可信控制面：冻结 RunManifest + QualificationManifest + BudgetPolicy
  内循环：ask → train → evidence/verdict → attribution → propose
            → screen → guard → apply → paired holdout → gate → lineage → tell
  结束/检查点：冻结候选与完整证据链
  外循环：propose → validate → shadow/replay → independent qualification
            → approve/promote 或 reject → 生成下一周期 manifest
最终报告：已锁定候选 → 一次 sealed evaluation → 报告，不回流搜索
```

一个周期内不得热替换 judge、rubric、collector、attributor、cell pins 或资格验证任务。
外循环每次先只改一个目标，其他组件冻结；多目标联合修改后续另立消融协议。
失败、拒绝、超时都留档；恢复执行必须核对 manifest，不能把不同实验缓存混用。

### 3.1 搜索与资格验证分离

- `search_env`：用于 train 和开发期 holdout；下一周期允许改变任务采样或环境格组合。
- `qualification_env`：独立、冻结的验证任务及格，外围候选不能选择、编辑或删减。
- 开发期 holdout 会反复影响选择，不能称为 sealed test。
- `sealed`：独立最终报告集，不参与提案、调参、筛选或外循环迭代；使用次数预先登记。
- 跨环境报告必须有逐格结果、覆盖率和配对差值，不能仅报改变任务分布后的均分。
- 如果搜索环境升级，旧/新系统均须在同一冻结资格环境重评；搜索分数不跨版本拼接。

## 4. 拟议数据契约

先实现为 runtime 内部模型和实验产物；确需跨进程调用时，再发布可选协议能力。
字段可以分文件存储，但 manifest 关联关系必须明确且可校验。

### 4.1 RunManifest（一次内循环的身份）

```text
run_id, schema_version, protocol_version, code_commit, seed
component_refs: slot → {name, version, artifact_hash, config_hash}
agent_genome, search_env_ref, qualification_manifest_hash
cells: cell_id → {axes_version, fingerprint, binding_hash}
judge_ref, rubric_hash, collector_ref, attribution_ref
split_manifest_hash, budget_policy_hash, security_policy_hash
```

配置哈希必须基于脱敏的规范配置；凭据不写入 manifest 或公开日志。
服务端模型版本不可钉住时明确记录该限制，不能声称严格可复现。

### 4.2 ExecutionBinding（声明到实际执行的桥）

```text
resolve(cell_id, fingerprint) -> BindingPlan
prepare(plan) -> ExecutionReceipt
run(task, sanitized_context, receipt) -> Trajectory
verify(receipt) -> BindingCheck
```

这是 runtime/harness 适配器契约，不擅自改变现有 `rsi_agent_run` 的必选参数。
`BindingPlan` 记录允许的镜像摘要、工作目录快照及 setup/agent/verifier 边界。
`ExecutionReceipt` 记录实际镜像/快照、探测结果、执行身份、阶段和 plan 哈希。
声明与实际不符必须停止；不支持绑定的执行器不能以“受控多环境”名义运行。
legacy 无格运行不受影响。`probes` 不能未经授权当 shell 命令直接执行。
MVP 使用预登记的安全探测器和有限本地 fixture，不实现通用远程环境管理器。

### 4.3 EvidenceRecord 与 DerivedRecord

```text
EvidenceRecord: record_id, run_id, episode_id, task_digest, cell_fingerprint,
  execution_receipt_hash, raw_trajectory_hash, raw_artifact_hashes, producer_ref
DerivedRecord: record_id, source_hashes, method_ref, method_config_hash,
  output_hash, created_at, supersedes?   # 新版本追加，不覆盖旧输出
```

原始任务、轨迹、工具输出和可信执行器观察是只追加记录；内容寻址并在读取时验哈希。
重算 verdict、证据解释或 attribution 生成新的 DerivedRecord，不能回写原始证据。
collector 升级若需新采集必须重跑且明确区分新 episode，不能伪装为旧轨迹的原始观察。
哈希只能发现篡改，不能证明写入者可信；真实对抗实验还需独立存储权限与可信采集边界。
缺证据、哈希不匹配、执行不完整分别报错，不把缺失分数当 0 分或通过。

### 4.4 OuterProposal 与 PromotionRecord

```text
OuterProposal: proposal_id, target_kind, base_ref, candidate_ref, patch_hash,
  hypothesis, source_record_hashes, expected_effect, validation_plan_hash
OuterValidation: proposal_id, frozen_manifest_hash, evaluator_ref,
  anchor_metrics, qualification_metrics, cost_metrics, checks, decision
PromotionRecord: proposal_id, validation_hash, approver, old_ref, new_ref,
  next_manifest_hash, status, rollback_ref
```

`target_kind` 首期仅接受明确白名单，禁止修改控制面、gate、sealed 数据和安全策略。
所有 candidate artifact、验证计划、依赖版本先冻结，再执行验证。
验证采用受信任 evaluator；候选 judge 只输出预测，由固定 evaluator 对照独立标签计算指标。
候选不能同时充当自身的评分者、标签生成者或准入决策者。
外循环提案器可看开发用错误分析，但不读 qualification/ sealed 的逐题标签或原始答案。
自动化首期止于生成验证报告；promotion 需显式审批，失败回滚 active pointer，记录仍保留。

## 5. MVP：让归因真正进入提案

在 PromptRewriter 中增加受限 attribution 上下文，复用已有 ProposalRequest，暂不新增协议字段。
按 `episode_id` 关联本轮 train 的 trajectory id；未知引用必须拒绝或明确标记丢弃原因。
输入包括 method、目标 kind/ref/weight/note 和可追溯证据引用，不能把相关性写成已证明因果。
反事实归因记录重跑次数、差值、环境指纹；它仍受采样噪声影响。
只允许本轮 train 内容进入提案；holdout/qualification/sealed 及其派生内容不得串入。
按稳定顺序截断条数和字符数；记录使用及截断的 attribution id/hash。
空归因保持现有行为；无失败题的行为保持兼容，不暗中扩大生成策略。
模型看到的归因是待分析数据，不是可执行指令；保留补丁护栏及防抄题检查。
验收必须检查真实请求载荷含归因，不以“跑过 explain()”代替消费者验证。

## 6. 环境格补充约束

继承 environment-cells.md 的 EnvCell、指纹算法、工具发现、坐标默认不暴露及 legacy 规则。
新增控制绑定不得改变环境与 harness/预算的概念边界。

1. 环境格声明在 run 开始冻结，每次取题及实际执行前核对；axes 版本单独冻结。
2. 相同 task id 可在多个格运行；内部使用 `(task_id, cell_id)`，外部键沿用 `<task_id>@<cell_id>`。
3. 为避免歧义，首版 cells 模式禁止 task id/cell id 含 `@`；legacy id 不受此限制。
4. 重复 tuple 直接报错，不能静默覆盖；lineage、gate、tell、导出全部使用一致键。
5. 父子比较须完全相同的 tuple 集、任务内容摘要、reference、环境与 judge 版本。
6. 每格任务可返回该格对应的标量/结构化 reference；暂不增加按格 reference 映射语法。
7. holdout 的 reference 只交可信 judge；coords 暴露开关不允许顺带暴露 reference/card 中的答案。
8. 非 digest pins 仍按原草案警告；严格资格验证模式拒绝不可固定的执行绑定。
9. 环境格模式必须在编排文件里写 `cells` 显式启用。不写 `cells` 时按 legacy 运行：即使 env 实现了 `rsi_env_cells`，也不枚举格、不要求 binding，旧 env 和旧 agent 都不受影响。写了 `cells` 才进入严格模式，每个格都必须有可验证的 binding，否则 fail fast。

本节解决实现所需的最小歧义；坐标 slug、GUI 坐标范围等仍由环境格草案另行决定。

## 7. 外围对象各自怎样验证

| 对象 | 开发信号 | 独立验证及不变量 |
|---|---|---|
| 场景/环境选择 | 搜索覆盖与失败分布 | 固定资格格上能力不退化；覆盖/成本按预注册统计 |
| judge/rubric | 开发 anchor 的错误分析 | 未暴露的 qualification anchor；一致率、误放率、误拒率，硬安全项不放松 |
| evidence 方法 | 已记录执行与采集缺口 | 可信 fixture 的来源真实性/完整性、篡改检测；旧 raw 记录字节不变 |
| attribution 方法 | 开发 episode 与干预 | 留出干预 fixture 的定位正确性、稳定性及成本；不把 agent 提分当唯一真值 |

不同 judge 下同一轨迹可产生不同判词，两个派生记录都保留。
阈值、样本量、允许回退幅度和成本上限在 validation plan 中预先锁定，不能看完结果再改。
增加 judge_epoch 本身不计入提升；跨周期结果统一回放固定判别器或在固定资格环境重跑。

## 8. 两条实验线

### A. RSIHub 外部组件替换

沿用 `benchmarks/rsihub_parity/` 和 `adapters/rsihub/`，先核对真实上游版本与安装。
对照 RSIHub 原版与只替换指定 OpenRSI 组件的插件版；逐项替换并保留消融。
模型、任务、种子、执行环境、评测器、总调用/token/时间预算一致，替换清单与版本入档。
受控对照的两组必须有相同可修改边界，固定 validate/gate；保留 Hyperagents 算子自进化的原版 parity 另列实验，不能混称固定边界实验。
严格模式显式锁定允许回退数、gate/guard 配置，不沿用未核对的默认值；父子完整逐题向量及任务 manifest 哈希必须一致。
逐题数据缺失或不匹配直接判验证失败，禁止适配器静默退回只比总分；文件护栏启用必需文件/受保护路径及编译检查。
用真实 SDK/适配器调用做负向探针（漏题、篡改 gate、改保护文件、缺文件、编译失败），不能只验证脚本入口或返回码。
此线回答组件替换效果，不证明原生 OpenRSI 的完整循环有效。
接口替身和 mock 只能证明契约；未跑真实 RSIHub 必须标记未验证。

### B. OpenRSI 原生完整循环

使用真实 Runner，保留 ask/train/attribution/propose/screen/guard/apply/holdout/gate/lineage/tell 事件。

| 组 | 策略 | 用途 |
|---|---|---|
| A | 冻结初始 agent，不更新 | 能力与成本基线 |
| B | 等搜索预算的独立候选，每次从同一根版本出发 | 排除“只是多试几次” |
| C | 同预算迭代，下一轮可继承已接受的改进 | 检验累积改进 |

B/C 的候选生成、训练评估、归因/反事实重跑、筛选与失败重试都计入搜索预算。
A 不必烧完搜索预算；明确报告各组实际消耗，并给予相同最终评估预算。
预先固定预算封顶、停止规则、种子数、候选选择规则；未完成任务和超时不得删行。
三组锁定最终候选后才各执行 sealed；不能用 sealed 选择每代最优。
初始可用小型合成 fixture 做集成 smoke，报告必须写“synthetic smoke，非 benchmark”。
报告逐题/逐格结果、成功率、配对差值、波动、接受/拒绝原因、调用/token/时长及成本缺测项。
外循环实验在内循环和环境绑定验收之后单独开启，不和首个 MVP 混跑。

## 9. 分阶段代码任务与验收

| 阶段 | 文件触点 | 完成条件 |
|---|---|---|
| P0 归因接入 | `components/llm/openrsi_llm/__init__.py`、新增 `tests/test_llm_attribution.py`、runtime 测试 | 归因实际出现在 train 提案载荷；空归因兼容；泄漏及未知引用测试 |
| P1 环境格协议 | `sdk/python/openrsi/{types,protocol,components,client,conformance}.py`、`runtime/openrsi_runtime/{config,wiring,loop}.py` | 格发现、过滤、指纹、tuple 分数、旧插件兼容贯通三种传输 |
| P2 受控绑定/记录 | 新增 `runtime/openrsi_runtime/{bindings,records}.py`，接入 harness、lineage | 两格 fixture 真正执行于不同绑定；receipt 可验；漂移停止；raw 不可覆盖 |
| P3 原生实验 | 新增 `benchmarks/native_loop/`，扩展 runtime 集成测试 | A/B/C、等预算 B/C、sealed 隔离、恢复一致与审计产物齐全 |
| P4 外围验证 | 新增 `runtime/openrsi_runtime/outer.py` 及版本化模型、judge 测试 | 单目标 proposal/validate/promote/reject/rollback；独立资格验证；默认关闭 |
| P5 外部实验 | `benchmarks/rsihub_parity/`、`adapters/rsihub/`、现场测试文档 | 原版/插件版真实运行或准确标记阻塞，不把 synthetic 当结果 |

新文件名是建议，允许按现有结构拆分；语义契约和测试不可省略。
P0 → P1 → P2 → P3 → P4 为依赖链；P5 可独立进行，但不替代 P3。
各阶段分代码 PR，不把未实现接口提前写入稳定协议或宣称支持。

### 必须通过的回归/负向测试

- [ ] 归因携带显著哨兵，确认出现在 proposer mock 请求；holdout 哨兵不得出现。
- [ ] 归因缺失/超长/无效引用均有明确处理；稳定截断可复现。
- [ ] 同题双格产生两条分数；重复 tuple、含 `@` 的新模式 id、未知格、错误 split 均失败。
- [ ] parent/candidate 的 tuple、任务摘要、reference 或执行指纹不一致时不得 gate。
- [ ] env 声明格正确但实际镜像/编码漂移时失败，证明不只是标签测试。
- [ ] 内循环中途改 judge/rubric/attributor/collector 被拒；恢复时 manifest 不一致被拒。
- [ ] 派生重算不会改变 raw 内容；篡改 raw 或 receipt 可检测；缺失证据不得默认通过。
- [ ] 候选“全部通过”的 judge 被独立负样本识别；不能修改 anchor 标签及安全硬项。
- [ ] 外围候选不读资格集标签；未获批准不得 promotion；拒绝/崩溃可恢复原 active ref。
- [ ] B 每轮根版本相同，C 确实继承接受版本；预算覆盖重跑和失败；sealed 不回流。
- [ ] 旧 0.4 无格配置、无新能力的 MCP 组件、已有 judge 共进化示例继续工作。

## 10. 协议发布与交付要求

P0 尽量不改 wire schema。P1 的环境格以拟议 0.5 可选能力发布，不能把草案当已发布协议。
工具发现作为能力协商；旧 env 不传 `cell`，旧 agent 不接收新必选参数。
新运行模式显式启用；不支持能力时 fail fast，不静默降级成标签式多环境实验。
外循环先作为 runtime 实验扩展，独立 schema_version；未来新增工具需单独协议审查。

实现者按 `CLAUDE.md` 执行 `pytest`、`ruff check .`。
修改 SDK types/protocol 后执行 `python -m openrsi.export_spec spec` 并核对生成文件无漂移。
每个阶段更新相关组件清单、REVIEW.md；修改 README 时中英文同步。
提交报告区分已通过、失败和未运行；真实模型/框架实验未跑，不得写成已兼容或已提升。
