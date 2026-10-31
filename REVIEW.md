# 审查清单

每次审查（代码审查、版本审查、PR 审查）都先回答下面三个问题，再看具体改动。回答要以仓库当前的代码为准，不要照抄下面的基线；基线只用来对比这次改动带来了什么变化。

## 三个问题

### 1. 我们的组件有哪些？

- 怎么查：`components/*/rsi-component*.yaml` 列出全部官方组件及其类型；`sdk/python/openrsi/protocol.py` 的 `KINDS` 列出协议定义的全部组件类型。
- 要回答：有哪些组件、各属于什么类型、哪些类型协议里有但还没有官方实现。
- 要检查：每个组件的清单和它实际暴露的工具一致（`tests/` 里有对应测试），并通过一致性测试（`openrsi-conformance`）。

### 2. 可自进化的组件有哪些？

- 怎么查：清单里 `kinds` 含 `evolvable` 的组件；运行时 `openrsi.yaml` 里 `components.memory` 和 `evolvables` 下配置的对象。
- 要回答：哪些东西能进化、各自怎么改（补丁格式）、怎么回滚、由谁把关（准入看 holdout，判别器看 anchor，护栏看补丁内容）。
- 要检查：进化是否守住了隔离规则（agent 看不到 reference 和 rubric、只有 train 进入提案、判别器修改只看 anchor、agent 和判别器轮流进化）。这些规则在 `runtime/openrsi_runtime/loop.py` 开头列着，都应有测试覆盖。

### 3. 是否能够兼容其他 RSI 框架？

- 怎么查：`adapters/` 下的适配器；组件能否作为普通 MCP server 被外部调用；`spec/` 里的协议是否足以让别的语言实现组件。
- 要回答：和哪些框架已经能互通、是双向还是单向、有没有用该框架的真实安装测过；哪些还只是计划。
- 要检查：不要把"计划"或"理论上可以"写成"已兼容"。没有用真实框架跑过的，明确写出来。

## v0.3 基线（2026-10）

| 问题 | 当时的答案 |
|---|---|
| 组件 | 协议定义 13 类：env、agent、evidence、judge、attribution、evolvable、memory、skill、proposer、algorithm、gate、guard、lineage。官方实现 11 个：evidence-judge、evolving-judge、memory、prompt、skills、rubric-evidence-attribution、lesson / claude / rubric-auditor / composite 提案器、greedy / archive 算法、holdout-gate、policy-guard、lineage。env 和 agent 没有官方实现，由用户提供（示例里有） |
| 可自进化 | memory（MemoryEvolvable）、prompt（PromptEvolvable）、skill 代码（SkillLibrary）、判别器的 rubric（EvolvingJudge）。都基于 SDK 的 ItemStore：补丁、快照树、回滚。可以同时进化多个。模型权重还不能进化 |
| 兼容其他 RSI | 所有组件都是 MCP server，任何支持 MCP 的框架都能调用。适配器：DSPy（判别器当 metric，含 GEPA 反馈）、LangGraph（消息转轨迹）。两者都只用假对象测过，没装真实的 dspy / langgraph 跑过。OpenEvolve、Letta、Mem0、TextGrad、Darwin Gödel Machine、verl 等只在项目文档里规划了，还没有代码 |

## v0.4 基线（2026-10）

| 问题 | 当时的答案 |
|---|---|
| 组件 | 协议仍是 13 类，新增可选工具 `rsi_algorithm_screen`。官方实现 13 个：在 v0.3 基础上加 harness（HarnessEvolvable + HarnessAgent，第一个官方 agent）和 weights（WeightsEvolvable，在 verl 适配器里）。算法从 2 种增加到 6 种：greedy、archive、pareto、shinka、islands、map-elites。gate 增加逐题不回退。env 仍没有官方实现 |
| 可自进化 | memory、prompt、skill 代码、判别器 rubric（都基于 ItemStore），加上 agent 代码仓库 harness（git 提交为快照，write / delete / diff 补丁，可改范围和受保护路径）和模型权重（checkpoint 为快照，一次训练为补丁，调用外部训练命令）。护栏能检查文件型补丁的受保护路径 |
| 兼容其他 RSI | MCP 通用。verl：判别器当 compute_score 奖励函数、RL / SFT 数据导出（拒绝非 train 数据）、权重可进化体；没用真实 verl 训练跑过。RSIHub：三个算子脚本（validate / gate / select）、归档导入；按 RSIHub 源码写，用接口替身测过，没在真实 RSIHub 里跑过。DSPy、LangGraph 同 v0.3，只用假对象测过 |
### v0.4 之后的变化

- verl 适配器：`tasks_to_rl_rows(for_validation=True)` 导出验证集并拒绝 train 题；只有 `expected` 的参考答案直接导出值（verl 自带奖励能读）；`WeightsEvolvable(model_glob=...)` 在 `global_step_N` 子目录里找模型
- RSIHub 适配器：准入先检查评估完整、可作父代（同 `parent_eligible`），拿不到可比的逐题分数时退回只比总分；护栏校验可要求 Python 文件能编译（`compile_globs`，和 hyperagents validate 一样）
- `field_tests/`：在真实 verl、RSIHub 上跑的现场测试；`benchmarks/rsihub_parity/`：用 RSIHub 评测器做的对照实验（Terminal-Bench 2 × MiniSWE × Hyperagents，baseline 对比换上 OpenRSI 的 validate 和 gate）。两者都还没有在真实环境跑过
- 新组件 `components/llm`：`ChatAgent`（OpenAI 兼容接口；上下文里有权重条目时按模型路径启动或重启推理服务）、`PromptRewriter`（LLM 改写系统提示的提案器），清单 `rsi-component.yaml`、`rsi-component.rewriter.yaml`
- `CompositeProposer` 的成员可以是任意提案器类；`output_equals` 可以先用正则抽取答案；`episode_messages` 保留 agent 自己的对话（含系统提示）
- `examples/small_model`：小模型同时进化系统提示和权重的全流程，有 vLLM、CPU（transformers）和假模型三种配置；假模型版本在 CI 里跑
- 修复：JSONL 读取改用按 `\n` 切分（`str.splitlines()` 会在 U+2028 等字符处断行，GSM8K 和 LLM 输出里都有），涉及谱系、anchor 集、RSIHub 归档、示例数据
- 组件化测试（`tests/test_composition.py`）：任意组合、三种传输方式结果一致、JavaScript 组件和客户端、故障与续跑、向前兼容；`openrsi-serve` 把任意 Python 组件类作为 MCP 服务运行；组件进程断开时报 `ComponentUnavailableError` 并指明组件和调用
- `PromptRewriter` 0.4.1 读归因（`spec/drafts/two-loop-evolution.md` 的 P0）：`ProposalRequest.attributions` 按 `episode_id` 对上展示的错题，附在该题下面，标明是数据不是指令；不属于本轮 train 的归因丢掉（`unknown_episode`），按权重排序后截断（`max_attribution_lines` / `max_attribution_chars`）；补丁说明记下用了哪几条（短哈希）和丢弃原因。没有归因时请求和以前完全一样。`tests/test_llm_attribution.py` 检查归因真的出现在发给 LLM 的请求里，包括经过 MCP 传输和完整循环
- 环境格（0.5 草案，`spec/drafts/environment-cells.md`，two-loop 设计的 P1）：SDK 0.4.1 加 `EnvCell`、可选工具 `rsi_env_cells`、`rsi_env_tasks` 的可选参数 `cell`（服务端拒绝未知格，没声明格的 env 收到 `cell` 也报错）、`LineageEntry.env_cells`；一致性测试对声明了格的 env 多查三项。运行时 0.4.1：编排文件写了 `cells` 才启用（`runtime/openrsi_runtime/cells.py`），每次取题前核对指纹、检查 `env_ref`、逐题分数键为 `<task_id>@<cell_id>`、轨迹 id 加 `@<cell_id>`、谱系记格、续跑必须同格，非 digest 钉住给警告；坐标默认不给 agent。协议版本仍是 0.4。只核对声明，不保证题目真在那个环境里执行（执行绑定是 P2，未做）。`tests/test_env_cells.py` 覆盖三种传输方式和各种必须停下的情况；不写 `cells` 的旧配置谱系不变
- 环境、取证、归因的配合：SDK / runtime 0.4.2，归因多一种粒度 `kind: env`（`ref` 为 `<坐标键>=<值>` 或 `cell:<格 id>`）；环境格模式下运行时给每个 episode 加一条 `env_cell` 证据（格、坐标、指纹，编排方记录，agent 看不到）。新组件 `CellContrastAttributor`（attribution 0.4.1，清单 `rsi-component.cell-contrast.yaml`）：同一道题有的格过、有的格挂时归到挂的格独有的坐标，全挂归到 `agent:capability`，篡改的环境证据不采信。`HoldoutGate` 0.4.1 加 `max_cell_regressions` / `cell_tolerance`（逐格不回退）。新组件 `RoutedProposer`（proposer 0.4.1，清单 `rsi-component.routed.yaml`）：按归因责任分流到成员提案器，算法点名优先，只在已配置成员里选。可进化体不变。`tests/test_cell_contrast.py`
