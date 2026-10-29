# OpenRSI

组件式的 agent 自我改进（RSI）框架和开放协议。自我改进的每个环节都是一个独立的 MCP server，任何框架都能单独拿去用，也能用一份 `openrsi.yaml` 把它们接成完整的循环。

当前版本 0.2：协议覆盖 11 类组件，提供 7 个官方组件、一个参考运行时、一致性测试，以及 DSPy 和 LangGraph 适配器。

## 自我改进循环

```
       ┌────────────── 进化算法选父快照 ◄──────────────┐
       ▼                                               │
  agent 在 train 题上做题 ──► 取证 ──► 判别 ──► 归因    │
                                                │      │
                                                ▼      │
                                   提案器提出 memory 补丁 │
                                                │      │
                                                ▼      │
                         候选快照在 holdout 上打分 ──► 准入 ──► 写入谱系
```

运行时保证三条规则，不依赖组件自觉：agent 看不到标准答案（`task.reference`）；只有 train 的结果会进入归因和提案；env 把题目划分标错就停止。

## 目录

```
spec/                     协议正文 PROTOCOL.md，以及生成的 JSON Schema 和工具定义
sdk/python/               openrsi-sdk：数据格式、11 类组件基类、MCP 服务端/客户端、一致性测试
runtime/                  openrsi-runtime：按 openrsi.yaml 接通组件，跑循环、续跑、查看谱系
components/
  judge/                  取证 + rubric 判别器（可选用 Claude 判断规则管不到的项）
  memory/                 可进化 memory：快照树、原子补丁、回滚、导出
  attribution/            沿判决引用的证据追责到轨迹步骤和 memory 条目
  proposer/               教训提案器（无需 LLM）和 Claude 提案器
  algorithm/              贪心 / 档案式开放进化
  gate/                   holdout 准入，可选显著性检验、anchor、成本上限
  lineage/                谱系，被拒绝的候选也留档
adapters/dspy/            判别器当 DSPy metric 用（含 GEPA 反馈）
adapters/langgraph/       LangGraph 状态转轨迹再判别
examples/                 能直接跑的例子
tests/                    64 个测试
```

## 快速开始：跑一个完整循环

```bash
pip install -e sdk/python -e runtime -e components/judge -e components/memory -e components/attribution \
            -e components/proposer -e components/algorithm -e components/gate -e components/lineage
cd examples/secret_ops
openrsi run                    # 跑到停为止；中断后再运行会从谱系接着跑
openrsi log                    # 查看谱系
```

例子里有 8 个符号，每个背后是一个未知运算。agent 从 memory 找规则，提案器从答错的 train 题里归纳规则。环境、执行体、提案器是自定义的，其余 6 个用官方组件；memory 作为独立的 MCP 子进程运行。实际输出：

```
 gen  parent                train    base    cand  result
  · root snap_root: holdout=0.000
   1  snap_root             0.111   0.000   0.375  接受 snap_7f0d3e1244746973  全部检查通过
   2  snap_7f0d3e124474697  0.667   0.375   0.500  接受 snap_7df638e7d96838e1  全部检查通过
   3  snap_7df638e7d96838e  0.000   0.500   0.875  接受 snap_f474c409fda4e93f  全部检查通过
   4  snap_f474c409fda4e93  1.000   0.875       -  提案器没有提出修改
  ...
   8  snap_f474c409fda4e93  0.778   0.875   1.000  接受 snap_0145c98ef4d8eda8  全部检查通过
  · 停止：达到目标分数 1.0
```

## openrsi.yaml

每个位置三种来源任选，可以混用：

```yaml
protocol: openrsi/0.2
name: my-experiment
plugins: [my_components]              # 相对本文件目录的 Python 模块
components:
  env:         {python: "my_components:MyEnv", with: {seed: 0}}         # 进程内
  agent:       {python: "my_components:MyAgent"}
  judge:       {python: "openrsi_judge:EvidenceJudge", with: {rubric: rubric.yaml, protect: ["tests/**"]}}
  evidence:    {same_as: judge}                                         # 一个组件担任两个位置
  memory:      {command: [openrsi-memory, --path, .openrsi/memory]}     # 子进程 MCP
  attribution: {python: "openrsi_attribution:RubricEvidenceAttributor"}
  proposer:    {python: "openrsi_proposer:ClaudeProposer"}              # 需要 anthropic 和 API 凭证
  algorithm:   {python: "openrsi_algorithm:ArchiveAlgorithm", with: {max_generations: 30, patience: 8}}
  gate:        {python: "openrsi_gate:HoldoutGate", with: {significance_z: 1.64}}
  lineage:     {url: "https://lineage.internal/mcp"}                    # 远程 MCP
loop: {retrieve_k: 5, max_patches: 2}
```

## 只用其中一个组件

每个组件都能单独用，不需要运行时：

```python
from openrsi_judge import EvidenceJudge
judge = EvidenceJudge({"items": [{"id": "tests_pass", "check": "tests_pass", "required": True},
                                 {"id": "no_test_edits", "check": "no_protected_edits", "required": True}]},
                      protect=["tests/**"])
verdict = judge.evaluate(task, trajectory, judge.collect(task, trajectory))
```

或者作为 MCP server 给任何客户端用（Claude Code、Cursor、其他语言的 agent）：

```bash
openrsi-judge --rubric rubric.yaml --protect 'tests/**'
openrsi-gate --significance-z 1.64
```

接进已有框架：

```python
from openrsi_dspy import judge_metric
optimizer = dspy.GEPA(metric=judge_metric(judge, to_task=..., to_trajectory=...), ...)
```

## 写自己的组件

继承 SDK 的基类，实现方法，用 `serve()` 暴露，再跑一致性测试：

```python
from openrsi import Gate, GateDecision, serve

class MyGate(Gate):
    name, version = "me/my-gate", "0.1.0"
    def check(self, request):
        ok = request.candidate["holdout"] > request.baseline["holdout"]
        return GateDecision(candidate_id=request.candidate_id, accepted=ok)

serve(MyGate())
```

```bash
openrsi-conformance -- python my_gate.py
openrsi-conformance --sample-task task.json -- python my_agent.py   # 领域相关的 agent / 提案器
```

其他语言不需要 SDK：实现 `spec/tools.json` 里的工具就行。

## 其他例子

```bash
python examples/judge_standalone.py          # 判别器抓住"改测试让测试变绿"的 agent
python examples/judge_standalone.py --mcp    # 同样的判断，走 MCP 协议
python examples/memory_evolution.py          # 不用运行时，手动把判别器和 memory 接成循环
```

## 开发

```bash
pip install -e sdk/python -e runtime -e components/* -e adapters/* pytest ruff
pytest
ruff check .
python -m openrsi.export_spec spec           # 改了数据格式后重新生成 spec（测试会检查它是否最新）
```

判别器的 `llm` 检查项和 `ClaudeProposer` 需要 `anthropic` 和 Anthropic API 凭证，默认模型 `claude-opus-5-5`，开启服务端拒答兜底（`fallbacks: "default"`）。其余组件都不需要网络。

## 下一步（v0.3）

skill、代码、工作流类型的可进化体和沙箱护栏；判别器本身参与进化（锚定人工标注集）；反事实归因；TypeScript SDK；OpenEvolve、Letta、OpenAI Agents SDK 适配器。
