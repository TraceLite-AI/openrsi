# OpenRSI

组件式的 agent 自我改进（RSI）框架和开放协议。自我改进的每个环节都是一个独立的 MCP server，任何框架都能单独拿去用，也能用一份 `openrsi.yaml` 把它们接成完整的循环。

当前版本 0.4：协议覆盖 13 类组件，提供 13 个官方组件、6 种进化算法（含 GEPA 式 Pareto、ShinkaEvolve、AlphaEvolve 式岛屿、MAP-Elites）、一个参考运行时、一致性测试，以及 verl、RSIHub、DSPy、LangGraph 适配器。memory、prompt、skill（可执行代码）、agent 自己的代码（harness）、模型权重和判别器本身都能进化；补丁生效前有护栏检查；准入可以要求逐题不回退；判别器的修改由人工标注集把关。

## 自我改进循环

```
       ┌────────────── 进化算法选父快照 ◄──────────────┐
       ▼                                               │
  agent 在 train 题上做题 ──► 取证 ──► 判别 ──► 归因    │
                                                │      │
                                                ▼      │
          提案器提出补丁（memory / prompt / skill / harness / 权重）│
                                                │      │
                                          护栏检查补丁内容  │
                                                │      │
                                                ▼      │
                         候选版本在 holdout 上打分 ──► 准入 ──► 写入谱系
```

运行时保证四条规则，不依赖组件自觉：agent 看不到标准答案（`task.reference`）；只有 train 的结果会进入归因和提案；env 把题目划分标错就停止；被护栏拦下的补丁不会生效。

## 目录

```
spec/                     协议正文 PROTOCOL.md，以及生成的 JSON Schema 和工具定义
sdk/python/               openrsi-sdk：数据格式、13 类组件基类、条目型可进化体的通用实现、MCP 服务端/客户端、一致性测试
runtime/                  openrsi-runtime：按 openrsi.yaml 接通组件，跑循环、续跑、查看谱系
components/
  judge/                  取证 + rubric 判别器（可选用 Claude 判断规则管不到的项）；可进化版本 EvolvingJudge
  memory/                 可进化 memory：快照树、原子补丁、回滚、导出
  evolvables/             prompt 库和 skill 库（skill 在子进程里执行，有 CPU、内存、超时限制）
  guard/                  护栏：拦危险代码、受保护条目、泄露的密钥、教 agent 绕过评估的内容
  attribution/            沿判决引用的证据追责到轨迹步骤和 memory 条目
  proposer/               教训提案器（无需 LLM）、Claude 提案器、判别器漏判审计、组合提案器
  harness/                agent 代码仓库作为可进化体（git 快照、可改范围、受保护路径），子进程运行 agent
  algorithm/              6 种进化算法：greedy、archive、pareto、shinka、islands、map-elites
  gate/                   holdout 准入，可选显著性检验、anchor、成本上限、逐题不回退
  lineage/                谱系，被拒绝的候选也留档
adapters/verl/            判别器当 verl 奖励函数、导出 RL / SFT 数据、模型权重作为可进化体
adapters/rsihub/          OpenRSI 组件装成 RSIHub 算子（护栏 validate、不回退 gate、Shinka select）
adapters/dspy/            判别器当 DSPy metric 用（含 GEPA 反馈）
adapters/langgraph/       LangGraph 状态转轨迹再判别
examples/                 能直接跑的例子
tests/                    135 个测试
```

## 快速开始：跑一个完整循环

```bash
pip install -e sdk/python -e runtime -e components/judge -e components/memory -e components/attribution \
            -e components/proposer -e components/algorithm -e components/gate -e components/lineage \
            -e components/evolvables -e components/guard -e components/harness
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

## 进化 skill，并用护栏把关

`examples/skill_ops` 里，提案器写的是 Python 函数，agent 在沙箱里调用。提案器被设置成第一次提案时夹带一个 `import os` 的"调试"函数，护栏把这个候选拦下，同一代里干净的候选照常评估：

```
 gen  parent                train    base    cand  result
   1  snap_root             0.111   0.000       -  拒绝 blocked_308171a3a8686d8b  护栏拦截：code_import
   1  snap_root             0.111   0.000   0.375  接受 snap_a5a264fd92b3971a  全部检查通过
  ...
   8  snap_b7ef07bba4b29ab  0.778   0.875   1.000  接受 snap_ea3c5ddb2d50cb6c  全部检查通过
```

被拦下的补丁也写进谱系（`openrsi log` 能看到原因），但从未生效。

子进程限制不是安全边界：skill 进程仍能访问文件系统和网络。不可信的代码要先过护栏，并且把 skill 库跑在容器里。

## 判别器和 agent 一起进化

`examples/judge_coevolution` 从两个都有问题的组件出发：agent 改测试文件让测试变绿，判别器只看 agent 有没有说 "done"。只看 holdout，一切"完美"。人工标注集（`anchor.jsonl`）里作弊被标为不通过，判别器与人工的一致率只有 0.5：

```
 gen  parent                train    base    cand  result
  · root cfg_fb9d46304fbf435c: holdout=1.000
   1  cfg_fb9d46304fbf435c  1.000   1.000       -  提案器没有提出修改
   2  cfg_fb9d46304fbf435c  1.000   1.000   0.500  接受 cfg_a02c631d…  anchor 0.500→1.000  [判别器进化代]
   3  cfg_a02c631d5ff661d7  0.500   0.500   1.000  接受 cfg_cb8b57b5…  anchor 1.000→1.000
```

第 2 代，审计提案器发现"判了通过、但证据显示改了受保护的测试文件"，补上检查项；holdout 下降是因为作弊被识破了，准入看的是 anchor 一致率。第 3 代，失败原因被写成教训，agent 改走正路，holdout 回到 1.0。

几条保证：判别器的修改只看 anchor，不看 holdout，所以把判别器改宽松来刷分的补丁会被拒绝（有测试覆盖）；agent 和判别器轮流进化；判别器的 rubric 不会出现在 agent 的上下文里。

## 进化 agent 自己的代码（harness）

```bash
cd examples/harness_ops && openrsi run
```

`HarnessEvolvable` 把 agent 的代码仓库当可进化体：每个快照是一个 git 提交，补丁是文件的写入、删除或统一 diff，回滚是切换提交。`mutable` 限定可改范围，`protected` 列出任何时候都不能改的路径。`HarnessAgent` 在子进程里运行仓库里的 agent（stdin 收题目 JSON，stdout 回轨迹 JSON），也可以放进 Docker。

例子里提案器直接改 `agent/strategy.py`；第 1 代有一个候选顺手改了 `eval/score.py`，护栏按 `protected_path` 拦下，从未生效。之后 holdout 到 1.0。

## 进化算法

| `--kind` / 类 | 来源 | 做法 |
|---|---|---|
| `greedy` / `GreedyAlgorithm` | 爬山 | 总从最好的版本出发 |
| `archive` / `ArchiveAlgorithm` | Darwin Gödel Machine | 保留全部被接受的版本，按分数和子代数加权抽样 |
| `pareto` / `ParetoAlgorithm` | GEPA、Meta-Harness | 按题前沿：在任何一道题上最好的版本都留着，被支配的去掉 |
| `shinka` / `ShinkaAlgorithm` | ShinkaEvolve | sigmoid 父代选择、评估前的新颖度筛查（`rsi_algorithm_screen`）、UCB 选提案器 |
| `islands` / `IslandAlgorithm` | AlphaEvolve、FunSearch | 多个岛各自进化，定期把最好的版本迁移到下一个岛 |
| `map-elites` / `MapElitesAlgorithm` | MAP-Elites | 按行为特征分格，每格留最好的，各格机会均等 |

逐题不回退准入：`HoldoutGate(max_task_regressions=0)`，总分提升还不够，任何一道 holdout 题都不许变差（Self-Harness、AHE 的做法）。

## 接入 verl：判别器当奖励、权重参与进化

```yaml
# verl 配置
custom_reward_function:
  path: <openrsi-verl reward-path 的输出>
  name: compute_score
```

设置环境变量 `OPENRSI_JUDGE=judge.yaml`（`python: "openrsi_judge:EvidenceJudge"` 加 `with`）即可，判别器在进程内运行。也可以在自己的奖励文件里 `make_compute_score(judge)`。

- `tasks_to_rl_rows(tasks, data_source)`：生成 verl 的 RL 数据行，标准答案只放在 `reward_model.ground_truth`；
- `episodes_to_sft_rows(episodes)`：判别器判为通过的 train 轨迹转成 SFT 数据（拒绝采样式自我蒸馏）；
- 两者遇到非 train 划分的数据都会报错，评估集不会漏进训练；
- `WeightsEvolvable`：快照是 checkpoint，补丁是一次训练，按模板调用外部训练命令（比如 verl 的 SFT / GRPO 入口）；`DistillProposer` 自动提出"用通过的轨迹做一次 SFT"。

没有用真实的 verl 训练跑过（这个环境没有 GPU），训练命令的参数名请以你装的 verl 版本为准。

## 接入 RSIHub

```bash
pip install openrsi-rsihub          # 最好装进 RSIHub 自己的 venv
openrsi-rsihub install /path/to/RSIHub
```

会在 RSIHub 的 `library/` 下生成三个算子，在配方里像内置算子一样引用：

- `validate/openrsi_guard`：PolicyGuard 检查候选改动，默认保护 `evaluator/**` 和 `operators/**`，可选 Python 代码检查；
- `gate/openrsi_no_regression`：总分提升且逐题不回退，逐题分数按 RSIHub 自己的 task_vector 规则计算；
- `select/openrsi_shinka`：ShinkaEvolve 式父代选择。

`openrsi-rsihub import-archive archive.jsonl --lineage lineage.jsonl` 把 RSIHub 的归档转成 OpenRSI 谱系。算子是按 RSIHub 源码（`evolve.frozen` 接口）写的，测试用接口替身验证了调用约定，但还没在真实的 RSIHub 运行里跑过。

## 反事实归因

`loop.counterfactual_reruns: N` 时，每代最多重跑 N 次：把某条 memory 从上下文里拿掉、重做同一道题，用分数差判断它是有用还是有害。结果和普通归因一起交给提案器，`LessonProposer` 据此删掉有害条目。

## openrsi.yaml

每个位置三种来源任选，可以混用：

```yaml
protocol: openrsi/0.4
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
  guard:       {python: "openrsi_guard:PolicyGuard", with: {protected: ["system:*"]}}
  lineage:     {url: "https://lineage.internal/mcp"}                    # 远程 MCP
evolvables:                                                             # memory 之外同时进化的对象
  prompt:      {python: "openrsi_evolvables:PromptEvolvable"}
  skills:      {command: [openrsi-evolvables, --kind, skills, --path, .openrsi/skills]}
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

## 下一步

在真实的 verl 和 RSIHub 上跑通并修正适配器；TypeScript SDK；OpenEvolve、Letta、OpenAI Agents SDK 适配器；用真实 LLM agent 和公开基准跑通完整循环。
