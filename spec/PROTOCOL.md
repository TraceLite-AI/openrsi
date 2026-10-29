# OpenRSI Protocol 0.1（草案）

OpenRSI 定义 agent 自我改进链条上可替换组件的接口。传输层用 [MCP](https://modelcontextprotocol.io)：一个组件就是一个 MCP server，暴露本文规定名字和参数的工具。本协议只规定 MCP 之上的语义。

机器可读的部分由参考实现生成，以它们为准：

- `schemas/*.json`：数据格式的 JSON Schema
- `tools.json`：每个工具的名字、所属类型、是否必选、输入输出 schema

文中"必须""不得"是规范要求，"应当"是强烈建议。1.0 之前可能有不兼容修改。

## 1. 组件与类型

一个组件实现一个或多个**类型**。v0.1 定义四种：

| 类型 | 作用 | 必选工具 |
|---|---|---|
| `evidence` | 从轨迹和环境收集可核验的证据 | `rsi_evidence_collect`、`rsi_evidence_verify` |
| `judge` | 基于证据给出判决 | `rsi_judge_evaluate` |
| `evolvable` | 被进化的对象，所有修改走补丁并生成快照 | `rsi_evolvable_describe`、`_snapshot`、`_apply`、`_rollback`、`_diff` |
| `memory` | 可检索的 evolvable | evolvable 的全部工具 + `rsi_memory_retrieve` |

所有组件必须实现 `rsi_component_info`，返回名字、版本、协议版本（`openrsi/0.1`）、类型和工具列表。调用方应当用它（或工具列表）判断组件能力，按能力降级。

可选工具：`rsi_judge_calibrate`。

## 2. 数据隔离

`Task.split` 取值 `train`、`holdout`、`anchor`、`final`。

- 只有 `train` 的轨迹和判决可以进入归因、提案等改进环节。`holdout`、`anchor`、`final` 只能用于打分和准入。
- `final` 应当只在发版时使用一次。
- 编排方负责执行这条规则；组件不得要求调用方提供 holdout 数据用于改进。

## 3. 证据

- `Evidence.sha256` 必须是 `content` 的规范 JSON（键排序、无空白、UTF-8）的 SHA-256。
- `id` 应当由类型、来源和哈希确定，使同样的输入得到同样的 id。
- 证据不可变。`rsi_evidence_verify` 必须拒绝内容与哈希不符的证据。
- 只有 `uri` 没有 `content` 的大对象，由持有者按 uri 复核。

## 4. 判决

- `Verdict.score` 在 0 到 1 之间。
- 每条 `rubric` 结论（pass 或 fail）必须引用证据 id 或给出说明；引用的 id 必须来自调用方提供的证据，或判别器在 `Verdict.collected` 里返回的证据。
- 判别器不得采信哈希不符的证据，应当在 `notes` 里说明。
- agent 输出里的自我评价不是证据。

## 5. 可进化体

- 所有修改必须通过 `rsi_evolvable_apply(patch)`；每次成功的 apply 生成一个新快照并设为当前快照，新快照的 `parent_ids` 包含原快照。
- `Patch.base_snapshot` 给出时，若与当前快照不同，必须拒绝（返回错误），且状态不变。
- 一个补丁要么全部生效，要么全部不生效。
- `rsi_evolvable_rollback(id)` 把当前快照切回任意已知快照，历史不得删除。快照组成一棵树。
- `rsi_evolvable_describe` 应当在 `example_ops` 里给出一组无害的示例操作，供一致性测试使用。
- `Patch.ops` 的格式由可进化体类型决定。memory 的格式见 `schemas/MemoryOp.json`：`add`（条目不能已存在）、`replace`（条目必须存在）、`remove`（条目必须存在）。

## 6. 错误

参数不合法、补丁冲突、找不到快照等调用方可以修正的错误，必须以 MCP 工具错误（`isError: true`）返回，并带上原因。组件内部故障可以只返回通用错误。

## 7. 组件清单

每个组件应当附带 `rsi-component.yaml`：

```yaml
protocol: openrsi/0.1
name: openrsi/evidence-judge
version: 0.1.0
kinds: [evidence, judge]
implements: [rsi_component_info, rsi_evidence_collect, rsi_evidence_verify, rsi_judge_evaluate, rsi_judge_calibrate]
runtime: {transport: stdio, command: [openrsi-judge, --rubric, rubric.yaml]}
evolvable_surfaces: [rubric]
```

清单声明的类型和工具必须与组件实际暴露的一致。

## 8. 一致性测试

参考实现提供与语言无关的一致性测试，通过 MCP 检查任意组件：

```bash
openrsi-conformance -- <启动组件的命令>
openrsi-conformance --url http://host/mcp
```

通过全部检查的组件可以标注"OpenRSI 0.1 兼容"。evolvable 检查会 apply 一个探针补丁再回滚，应当对临时实例运行。

## 9. 预留

`Attribution`、`GateDecision` 的格式已在 schema 中给出，对应的归因、准入组件类型计划在 0.2 定义。
