# OpenRSI

组件式的 agent 自我改进（RSI）框架和开放协议。每个环节都是一个独立的 MCP server，任何框架都能单独拿去用。

这是 v0.1：协议草案、Python SDK、两个旗舰组件（基于证据的判别器、可进化 memory）、一致性测试、DSPy 和 LangGraph 适配器。

## 目录

```
spec/                      协议正文 PROTOCOL.md，以及生成的 JSON Schema 和工具定义
sdk/python/                openrsi-sdk：数据格式、组件基类、MCP 服务端/客户端、一致性测试
components/judge/          openrsi-judge：取证 + rubric 判别器（可选用 Claude 判断规则管不到的项）
components/memory/         openrsi-memory：可进化 memory，快照树 + 回滚
adapters/dspy/             openrsi-dspy：判别器当 DSPy metric 用（含 GEPA 反馈）
adapters/langgraph/        openrsi-langgraph：LangGraph 状态转轨迹再判别
examples/                  两个能直接跑的例子
tests/                     37 个测试
```

## 三种用法

**1. 在 Python 里直接用一个组件**

```python
from openrsi import from_messages, Task
from openrsi_judge import EvidenceJudge, Rubric

judge = EvidenceJudge(Rubric.model_validate({"items": [
    {"id": "tests_pass", "check": "tests_pass", "required": True},
    {"id": "no_test_edits", "check": "no_protected_edits", "required": True},
]}), protect=["tests/**"])

traj = from_messages(openai_style_messages, task_id="fix-123")
verdict = judge.evaluate(Task(id="fix-123"), traj, judge.collect(Task(id="fix-123"), traj))
```

**2. 作为 MCP server 给任何框架用**

```bash
openrsi-judge --rubric rubric.yaml --protect 'tests/**'
openrsi-memory --path ./memory-store
```

任何 MCP 客户端（Claude Code、Cursor、其他语言的 agent）都能直接调用 `rsi_judge_evaluate`、`rsi_memory_retrieve` 等工具。Python 里：

```python
from openrsi import connect
async with connect(["openrsi-judge", "--rubric", "rubric.yaml"]) as c:
    verdict = await c.judge.evaluate(task, traj)
```

**3. 接进已有框架**

```python
from openrsi_dspy import judge_metric
optimizer = dspy.GEPA(metric=judge_metric(judge, to_task=..., to_trajectory=...), ...)
```

## 例子

```bash
python examples/judge_standalone.py         # 抓住"改测试让测试变绿"的 agent
python examples/judge_standalone.py --mcp   # 同样的判断，走 MCP 协议
python examples/memory_evolution.py         # 判别器 + memory 的最小自我改进循环
```

`memory_evolution.py` 的输出（holdout 正确率从 0 升到 1.0，没有提升的候选会被回滚）：

```
round  train  holdout   cand  result
    1   0.11     0.00   0.38  accepted → snap_7f0d3e1244746973
    2   0.67     0.38   0.50  accepted → snap_7df638e7d96838e1
    3   0.00     0.50   0.88  accepted → snap_f474c409fda4e93f
  ...
    8   0.78     0.88   1.00  accepted → snap_0145c98ef4d8eda8
```

## 写自己的组件

继承 SDK 里的基类，实现方法，用 `serve()` 暴露：

```python
from openrsi import Judge, Verdict, RubricResult, serve

class MyJudge(Judge):
    name, version = "me/my-judge", "0.1.0"
    def evaluate(self, task, trajectory, evidence):
        ...
        return Verdict(score=..., passed=..., judge_ref=self.ref, rubric=[...])

serve(MyJudge())
```

然后跑一致性测试：

```bash
openrsi-conformance -- python my_judge.py
```

其他语言不需要 SDK：实现 `spec/tools.json` 里的工具就行，一致性测试通过 MCP 检查。

## 开发

```bash
pip install -e sdk/python -e components/judge -e components/memory -e adapters/dspy -e adapters/langgraph pytest ruff
pytest                                     # 全部测试
ruff check .
python -m openrsi.export_spec spec         # 改了数据格式后重新生成 spec（测试会检查它是否最新）
```

判别器的 `llm` 检查项需要 `pip install "openrsi-judge[llm]"` 和 Anthropic API 凭证，默认模型 `claude-opus-5-5`，并开启服务端拒答兜底（`fallbacks: "default"`）。其余检查不需要网络。

## 下一步（v0.2）

归因、提案器、进化算法、准入、谱系五类组件，以及编排它们的运行时和 `openrsi.yaml`。见项目文档的路线图。
