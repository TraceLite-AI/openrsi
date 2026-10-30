# 草案：环境格与环境坐标（拟入 0.5）

状态：讨论稿，未实现。参考实现、`schemas/`、`tools.json` 都还没改。和 PROTOCOL.md 一样，"必须""不得"是规范要求，"应当"是强烈建议。

## 为什么要补

0.4 的 `env` 只管出题。题目在什么环境里跑，协议没有说法：`Task.env_ref` 字段存在，但没有规定含义，参考实现也不读它。这会带来两个问题：

1. **准入可能比错。** 准入靠比较父版本和候选在 holdout 上的分数。如果两次评估之间环境悄悄变了（镜像按 tag 拉到了新版本、时区或编码不同），分数差就不能归到补丁上。
2. **没法研究"改进能不能跨环境"。** 在一个环境里学到的 memory、skill、harness 改动，换个环境还管不管用，现在无从记录，也无从比较。

本草案借用 [Awesome-Agentic-Environments](https://github.com/TraceLite-AI/Awesome-Agentic-Environments) 的环境定义和坐标体系（`docs/environment-axes.md` 定稿草案 v1、`docs/environment-card.md` v0.1），不另起一套。

## 1. 定义

- **环境**：被测策略之外、能改变任务正确解的一切。harness（工具定义、上下文管理、步数上限）和评判、预算不是环境。
- **坐标**：环境在某个子维度上的取值，比如 `E2.1:os = windows`、`E4.1:默认编码 = cp1252`。子维度和取值以 environment-axes 为准。
- **钉住清单**：复现环境需要的具体版本，比如镜像摘要、包版本策略、快照 id。
- **环境格**（cell）：一组坐标加一份钉住清单，代表一个可以复现的环境。一道题可以在多个格里跑；"同一道题、只换环境"就是同一个 task id 配不同的格。

## 2. 数据格式

新增 `EnvCell`：

```json
{
  "id": "linux-cp1252",
  "axes": "aae-environment-axes/1",
  "coords": {
    "E2.1:os": "linux",
    "E2.2:isolation": "container",
    "E4.1:默认编码": "cp1252"
  },
  "stage": {"E5.2:network": {"setup": "online", "agent": "offline", "verifier": "offline"}},
  "pins": {"image": "envbench-base@sha256:b499edbd…", "pin_policy": "digest"},
  "card": "https://example.org/cards/linux-cp1252.md",
  "probes": {"E4.1:默认编码": "python -c 'import locale;print(locale.getpreferredencoding())'"}
}
```

| 字段 | 含义 |
|---|---|
| `id` | 格的 id，在一个 env 组件内唯一 |
| `axes` | 坐标体系及版本。本草案只定义 `aae-environment-axes/1` |
| `coords` | 坐标。键是"维度编号:子维度名"，值取自 axes 文档。**没写的键表示"未声明"，不表示"取基准值"**（与环境卡片规则 7 一致） |
| `stage` | 可选。按阶段（setup / agent / verifier）取值不同的坐标，主要是 E5 边界层 |
| `pins` | 钉住清单。`pin_policy` 取 `digest`、`tag`、`none` |
| `card` | 可选。完整环境卡片的地址 |
| `probes` | 可选。坐标的观测命令（axes 第 4 节的"读数"），供编排方或一致性测试核对 |

`Task.env_ref` 的含义改为：**这道题所在格的 `id`**。为空表示未声明，行为和 0.4 一样。

格的**指纹**是 `coords`、`stage`、`pins` 的规范 JSON（键排序、无空白、UTF-8）的 SHA-256，和证据哈希的算法相同。

## 3. 工具

- 新增可选工具 `rsi_env_cells()`：返回这个 env 能提供的全部格（`list[EnvCell]`）。
- `rsi_env_tasks` 增加可选参数 `cell`：给了就只出这个格里的题。不认识的格必须返回工具错误。
- `rsi_component_info` 不变。调用方看工具列表里有没有 `rsi_env_cells`，判断 env 是否支持环境格。

实现了 `rsi_env_cells` 的 env，返回的每道题的 `env_ref` 都必须是它声明过的格 id。

## 4. 规则

这一节是对第 2 节"数据隔离"的补充，编排方必须执行。

1. **格不能中途变。** 编排方在运行开始时记下每个格的指纹，之后每次取题都要核对。指纹变了必须停止，和发现划分错误一样处理。
2. **准入只在同一格上比。** 父版本和候选的 holdout 必须是同一批 (task id, 格) 组合。逐题分数的键在有格时写成 `<task_id>@<cell_id>`，没有格时仍是 task id。
3. **钉住应当用摘要。** `pin_policy` 不是 `digest` 时，编排方应当在谱系里留警告：同一个 tag 两次拉到的可能不是同一个镜像。
4. **坐标默认不给 agent。** 交给 agent 的任务保留 `env_ref`，但不附带 `coords` 和 `card`。agent 需要知道环境时应当自己探测，和真实部署一致。实验要测"告诉 agent 环境会不会更好"时，在编排文件里显式打开（见第 6 节）。
5. **谱系记录格。** `LineageEntry` 增加 `env_cells`：这次评估用到的格 id 到指纹的映射。
6. **跨格泛化**（可选的实验设置）：编排文件可以让 train 和 holdout 用不同的格（比如 train 用 `linux-utf8`，holdout 用 `linux-cp1252`）。这时 holdout 的分数应当按格分别报告，不只报总分。

## 5. 兼容性

- 只做增加：0.4 的 env 不实现 `rsi_env_cells` 也照样能用，`env_ref` 仍可以是任意字符串，编排方不检查。
- 环境格只是描述，不是隔离手段。真正把题放进某个环境里跑，仍然是 agent 或 harness 的事（比如 HarnessAgent 的 `docker_image`）。
- 坐标体系换版本时（`axes` 变了），不同版本的格不得混在同一次准入里比较。

## 6. 编排文件

```yaml
components:
  env: {python: "envshift_openrsi:EnvShiftEnv", with: {suite: coding}}
cells:
  train:   [linux-utf8]
  holdout: [linux-utf8, linux-cp1252]   # 同一批题，两个格都考
  expose_coords_to_agent: false          # 规则 4 的开关，默认 false
```

不写 `cells` 时，编排方用 env 给出的全部格。

## 7. 对照：几个已有环境怎么映射

| 环境 | 格怎么划 |
|---|---|
| EnvShift v1 | 每个系统一个格：`E2.1:os` 取 linux / macos / windows，`pins` 写 GitHub runner 标签（tag 级，按规则 3 留警告） |
| EnvShift v2 | 一个基准格加若干变体格，每个变体只改一个子维度；`pins.image` 用 `envbench-base` 的摘要 |
| Terminal-Bench / SWE-bench | 每题一个格，坐标大多是"未声明"；环境卡片已有人工填写的版本可以直接引用 |

## 8. 还没定的

1. 坐标的键要不要改成不含中文的 slug（比如 `E4.1:encoding`）。axes 文档的子维度名有中有英，跨语言实现时可能不方便。
2. 编排方要不要自己跑 `probes` 核对坐标。跑能防止声明和实际不符，但需要编排方进得了 agent 所在的环境。
3. GUI 任务的 E7 界面层坐标（分辨率、输入法）是否也要求钉住。
4. 一道题在不同格里的正确解可能不同。`reference` 要不要允许按格给（`reference` 是一个以格 id 为键的对象）。

## 9. 落地要改的地方

- `sdk/python/openrsi/types.py`：加 `EnvCell`，给 `LineageEntry` 加 `env_cells`；`protocol.py`：加 `rsi_env_cells`，给 `rsi_env_tasks` 加 `cell`；然后 `python -m openrsi.export_spec spec`。
- `runtime/`：运行开始时记录指纹，取题时核对；逐题分数的键带格；读编排文件的 `cells`。
- `conformance`：检查 `env_ref` 都是声明过的格，`cell` 参数能过滤，未知格报错。
- PROTOCOL.md 加一节，README 两版同步，REVIEW.md 基线记下变化。
