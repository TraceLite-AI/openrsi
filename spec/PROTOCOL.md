# OpenRSI Protocol 0.4（草案）

OpenRSI 定义 agent 自我改进链条上可替换组件的接口。传输层用 [MCP](https://modelcontextprotocol.io)：一个组件就是一个 MCP server，暴露本文规定名字和参数的工具。本协议只规定 MCP 之上的语义。

机器可读的部分由参考实现生成，以它们为准：

- `schemas/*.json`：数据格式的 JSON Schema
- `tools.json`：每个工具的名字、所属类型、是否必选、输入输出 schema

文中"必须""不得"是规范要求，"应当"是强烈建议。1.0 之前可能有不兼容修改。0.2、0.3、0.4 都只做了增加，之前版本的组件不用改就符合当前版本。

## 1. 组件与类型

一个组件实现一个或多个**类型**。0.4 定义 13 种：

| 面 | 类型 | 作用 | 必选工具 |
|---|---|---|---|
| 执行 | `env` | 按划分出题 | `rsi_env_tasks` |
| 执行 | `agent` | 在给定 memory 上下文中做题 | `rsi_agent_run` |
| 评估 | `evidence` | 收集可核验的证据 | `rsi_evidence_collect`、`rsi_evidence_verify` |
| 评估 | `judge` | 基于证据判决 | `rsi_judge_evaluate` |
| 评估 | `attribution` | 说明成败归因于什么 | `rsi_attribution_explain` |
| 进化 | `evolvable` | 被进化的对象，修改走补丁、生成快照 | `rsi_evolvable_describe`、`_snapshot`、`_apply`、`_rollback`、`_diff` |
| 进化 | `memory` | 可检索的 evolvable | evolvable 的全部工具 + `rsi_memory_retrieve` |
| 进化 | `skill` | 条目是可执行代码的 evolvable | evolvable 的全部工具 + `rsi_skill_invoke` |
| 进化 | `proposer` | 根据 train 结果提出补丁 | `rsi_proposer_propose` |
| 进化 | `algorithm` | 选父快照、决定何时停 | `rsi_algorithm_ask`、`rsi_algorithm_tell` |
| 进化 | `gate` | 决定候选能否被接受（看分数） | `rsi_gate_check` |
| 进化 | `guard` | 补丁生效前检查内容（看补丁本身） | `rsi_guard_inspect` |
| 基础 | `lineage` | 记录快照、分数和准入决定 | `rsi_lineage_record`、`rsi_lineage_log`、`rsi_lineage_best` |

所有组件必须实现 `rsi_component_info`，返回名字、版本、协议版本、类型和工具列表。调用方应当用它（或工具列表）判断组件能力，按能力降级。

可选工具：`rsi_judge_calibrate`、`rsi_evolvable_export`、`rsi_algorithm_screen`（0.4）、`rsi_env_cells`（0.5 草案，见第 14 节）。

## 2. 数据隔离

这一节是协议的核心约束，编排方（比如参考运行时）必须执行。

- `Task.split` 取值 `train`、`holdout`、`anchor`、`final`。只有 `train` 的 episode 可以交给 `attribution` 和 `proposer`。`holdout`、`anchor`、`final` 只用于打分和准入。`final` 应当只在发版时使用一次。
- `Task.reference` 放标准答案等只给判别器看的信息。交给 `agent` 的任务必须去掉 `reference`。
- `env` 必须给每道题标对 `split`；同一轮里 train 和 holdout 的题目 id 不得重叠。编排方发现划分错误时必须停止。
- agent 输出里的自我评价不是证据。

## 3. 证据

- `Evidence.sha256` 必须是 `content` 的规范 JSON（键排序、无空白、UTF-8）的 SHA-256。
- `id` 应当由类型、来源和哈希确定，使同样的输入得到同样的 id。
- 来自轨迹某一步的证据，`source` 应当写成 `trajectory:<trajectory_id>:step:<step_id>`，归因据此追到步骤。
- 证据不可变。`rsi_evidence_verify` 必须拒绝内容与哈希不符的证据。

## 4. 判决

- `Verdict.score` 在 0 到 1 之间。
- 每条 rubric 结论（pass 或 fail）必须引用证据 id 或给出说明；引用的 id 必须来自调用方提供的证据，或判别器在 `Verdict.collected` 里返回的证据。
- 判别器不得采信哈希不符的证据，应当在 `notes` 里说明。

## 5. 归因

- `rsi_attribution_explain` 对每个 episode 返回一条 `Attribution`，`episode_id` 等于 `trajectory.id`。
- `AttributionTarget.weight` 在 -1 到 1 之间：负数是责任，正数是功劳。`kind` 取 `step`、`item`、`component`、`patch`、`rubric`，以及 `env`（0.5 草案，见第 14 节）。
- 失败的 episode 应当至少有一个负权重目标。

## 6. 可进化体

- 所有修改必须通过 `rsi_evolvable_apply(patch)`；成功的 apply 生成新快照并设为当前快照，新快照的 `parent_ids` 包含原快照。
- `Patch.base_snapshot` 给出时，若与当前快照不同，必须拒绝且状态不变。一个补丁要么全部生效，要么全部不生效。
- `rsi_evolvable_rollback(id)` 切回任意已知快照，历史不得删除。快照组成一棵树。
- `rsi_evolvable_describe` 应当在 `example_ops` 里给出一组无害的示例操作，供一致性测试使用。
- `rsi_evolvable_export(snapshot_id)` 可选，返回快照的完整内容，提案器据此决定 add 还是 replace。
- memory 的 ops 格式见 `schemas/MemoryOp.json`：`add`（条目不能已存在）、`replace`（必须存在）、`remove`（必须存在）。
- `rsi_memory_retrieve`：`tags` 是过滤条件，`query` 只用于排序。
- **文件型可进化体**（0.4，`type: harness`，比如 agent 的代码仓库）的 ops：`{"op": "write", "path", "content"}`、`{"op": "delete", "path"}`、`{"op": "diff", "unified"}`（统一 diff 格式）。路径是仓库内的相对路径，不得包含 `..`。可进化体应当声明可改范围和受保护路径，越界的补丁必须整个拒绝；`export` 返回可改范围内的文本文件，条目 id 是路径。
- **权重型可进化体**（0.4，`type: weights`）：快照是一个 checkpoint，补丁是一次训练 `{"op": "train", "method", "data", "config"}`，每个补丁只能有一个 train。回滚是切换 checkpoint，旧 checkpoint 不删。`export` 返回一条 `kind: weights` 的条目，`text` 是模型路径。训练数据只能来自 train 划分（参考实现导出非 train 数据时报错）。

## 7. 提案、算法、准入、谱系

- **提案器**：返回的补丁数不得超过 `max_patches`；`target` 必须等于请求里的目标；`base_snapshot` 为空或等于 `parent_snapshot`。
- **进化算法**：第一次 `tell` 登记根快照（`accepted: true`，带指标）。`ask` 只能返回被接受过的快照，返回 `parent_id: null` 表示停止并给出原因。同一代可以 `tell` 多次（多个候选）。
- **逐题分数**（0.4）：编排方应当在 `TellRequest.task_scores`、`GateRequest.baseline_tasks / candidate_tasks` 和 `LineageEntry.task_scores` 里给出每道 holdout 题的分数（键是 task id）。按题前沿（GEPA 式 Pareto）和逐题不回退准入依赖它。
- **评估前筛查**（0.4，可选）：算法实现 `rsi_algorithm_screen(patch, parent_id)` 时，编排方必须在评估候选之前调用；`accept: false` 的补丁不评估、不 apply，按拒绝 `tell`。用于新颖度过滤（ShinkaEvolve 式），省掉重复候选的评估成本。
- **提案器提示**（0.4）：`AskResult.proposer` 是算法建议这一代用的提案器名字，编排方放进 `ProposalRequest.hints.proposer`；组合提案器应当只用这个成员，并在 `proposer_ref` 末尾加 `#成员名`。编排方在 `TellRequest.proposer` 里回报产生候选的提案器（包括没产出补丁、被筛掉、被拦下的情况），算法据此做自适应选择。
- **准入**：比较 `baseline`（父快照）和 `candidate` 的指标，返回每项检查的结果。明显变差的候选不得接受。参考实现可选"逐题不回退"：分数下降的题数超过上限就拒绝（Self-Harness、AHE 的做法）。
- **谱系**：`record` 同一个 id 再次记录时以最后一次为准。`best` 只在 `root` 和 `accepted` 里选，被拒绝的候选不算。

## 8. skill 与护栏（0.3）

- **skill**：条目的 `data.code` 是源码，`data.entry` 是入口函数名（默认 `run`）。`apply` 必须拒绝语法错误或没有入口函数的代码。`rsi_skill_invoke(skill_id, arguments, snapshot_id)` 运行指定快照里的 skill，返回 `SkillResult`；skill 抛出的异常放在 `error` 里，不作为工具错误；找不到 skill 才是工具错误。
- skill 应当在隔离环境里执行：独立进程、受限的 CPU 和内存、超时、空环境变量。子进程隔离不是安全边界，不可信代码还应当经过护栏，并运行在容器里。
- **护栏**：`rsi_guard_inspect(patch, target)` 返回 `GuardReport`。`allowed` 必须等于"没有 block 级别的发现"。编排方配置了护栏时，被拦下的补丁不得 apply，应当以 `rejected` 状态写进谱系（id 以 `blocked_` 开头），留作审计。warn 级别的发现不阻止补丁，但应当随准入决定一起留档。
- 护栏对文件型补丁（0.4）应当检查路径：改动受保护路径（比如评估脚本）必须拦下，这是防止 agent 改评估器刷分的主要手段。

## 9. 多个可进化体（0.3）

- 编排方可以同时进化多个可进化体。一个**版本**是每个可进化体各取一个快照的组合，`LineageEntry.snapshots` 记录 target 到快照 id 的映射。
- 只有一个可进化体时，版本 id 就是它的快照 id；多个时是组合的哈希（参考实现用 `cfg_` 前缀）。谱系的 `target` 是全部 target 名排序后用 `+` 连接。
- `ProposalRequest.targets` 和 `states` 列出全部可进化体及其父版本内容；补丁的 `target` 决定改哪一个。一个补丁只改一个可进化体。
- 交给 agent 的上下文：memory 按题目检索，其他可进化体（prompt、skill）导出全部条目。判别器自身（同时是 judge 和 evolvable 的组件）的内容不得交给 agent。

### 判别器参与进化

判别器的修改会改变打分标准：判别器变宽松，holdout 分数就会虚高。所以：

- 判别器的修改必须用人工标注集（anchor）验证。编排方用 `rsi_judge_calibrate` 计算与人工判断的一致率，记为指标 `anchor`；`GateRequest.primary_metric` 设为 `anchor`，判别器的修改只有在一致率提升时才被接受，与 holdout 怎么变无关。没有 anchor 集时，编排方不得接受判别器的修改。
- agent 侧和判别器必须轮流进化，同一代只改一边，分数变化才能归因到一方。
- 每接受一次判别器的修改，版本进入新的纪元（`judge_epoch` 加 1）。不同纪元的 holdout 不可直接比较；参考运行时用 `progress = 2 × judge_epoch + holdout` 选最佳版本，新纪元优先。

### 反事实归因

编排方可以把某个条目从上下文里拿掉、重跑同一道题，用分数差作为该条目的权重（`method: counterfactual`）：拿掉后分数下降说明它有用，上升说明它有害。这需要重跑 agent，代价较高，应当有预算上限。

## 10. 错误

参数不合法、补丁冲突、找不到快照等调用方可以修正的错误，必须以 MCP 工具错误（`isError: true`）返回，并带上原因。组件内部故障可以只返回通用错误。

## 11. 组件清单

每个组件应当附带 `rsi-component.yaml`，声明的类型和工具必须与组件实际暴露的一致：

```yaml
protocol: openrsi/0.4
name: openrsi/holdout-gate
version: 0.4.0
kinds: [gate]
implements: [rsi_component_info, rsi_gate_check]
runtime: {transport: stdio, command: [openrsi-gate]}
```

## 12. 一致性测试

参考实现提供与语言无关的一致性测试，通过 MCP 检查任意组件：

```bash
openrsi-conformance -- <启动组件的命令>
openrsi-conformance --url http://host/mcp
openrsi-conformance --sample-task task.json -- <命令>   # 领域相关的 agent / 提案器
```

通过全部检查的组件可以标注"OpenRSI 0.4 兼容"。evolvable、algorithm、lineage 的检查会写入数据，应当对临时实例运行。

## 13. 编排文件 openrsi.yaml

编排文件不属于组件协议，是参考运行时的约定，列在这里方便其他实现兼容：

```yaml
protocol: openrsi/0.4
name: my-experiment
plugins: [my_components]
components:
  env:      {python: "my_components:MyEnv", with: {seed: 0}}   # 进程内
  memory:   {command: [openrsi-memory, --path, .openrsi/memory]} # 子进程 MCP
  judge:    {url: "https://judge.example.com/mcp"}              # 远程 MCP
  evidence: {same_as: judge}                                    # 同一组件担任两个位置
  guard:    {command: [openrsi-guard]}
evolvables:                                                     # memory 之外参与进化的对象
  skills:   {command: [openrsi-evolvables, --kind, skills, --path, .openrsi/skills]}
  prompt:   {python: "openrsi_evolvables:PromptEvolvable"}
loop: {generations: 20, retrieve_k: 5, max_patches: 1}
```

必选位置：env、agent、judge、proposer、algorithm、gate、lineage。可选：memory、evidence、attribution、guard。memory 和 evolvables 至少要有一个。

## 14. 环境格（0.5 草案，可选）

完整定义见 [drafts/environment-cells.md](drafts/environment-cells.md)，这里只列已经进入 `schemas/` 和 `tools.json` 的部分。草案在 1.0 之前可能变。

- `EnvCell`：`id`（不得含 `@`）、`axes`、`coords`、`stage`、`pins`、`card`、`probes`。指纹是 `axes`、`coords`、`stage`、`pins` 的规范 JSON 的 SHA-256。
- `rsi_env_cells()` 可选，返回 `EnvCell` 列表。`rsi_env_tasks` 多一个可选参数 `cell`：给了只出这个格的题，题目的 `env_ref` 必须等于格 id；未知的格、或没声明格的 env 收到 `cell`，必须返回工具错误。不传 `cell` 时行为和 0.4 相同。
- 编排文件写了 `cells` 才进入环境格模式（第 13 节的格式之外加一节 `cells: {train, holdout, expose_coords_to_agent}`）。这时编排方必须：每次取题前核对指纹，变了就停止；检查 `env_ref`；逐题分数的键写成 `<task_id>@<cell_id>`，重复就停止；在 `LineageEntry.env_cells` 记下格 id 到指纹；续跑时格必须和谱系一致。默认不把坐标交给 agent。
- 环境格模式下编排方应当给每个 episode 加一条 `type: env_cell` 的证据，内容是格 id、坐标和指纹，来源写成 `runtime:cell:<格 id>`；它由编排方记录，不交给 agent。
- 归因可以用 `kind: env` 指向环境坐标：`ref` 写成 `<坐标键>=<值>`（如 `E4.1:默认编码=cp1252`），坐标不明时写 `cell:<格 id>`。参考实现 `CellContrastAttributor` 用同一道题在不同格里的结果做对比得出它；这是相关，不证明因果。
- 环境格只是声明，不是隔离手段。
