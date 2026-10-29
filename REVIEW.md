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
